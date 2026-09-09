# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for structured JSON logging and JsonLoggingPlugin."""

import io
import json
import logging
from unittest.mock import MagicMock

import pytest

from app.agent import route_user_request
from app.logging_config import (
    JsonFormatter,
    JsonLoggingPlugin,
    PiiRedactionFilter,
    redact_pii,
    redact_pii_string,
)
from app.tools import (
    check_availability,
    create_event,
    delete_event,
    list_events,
    update_event,
)


class TestJsonFormatter:
    """Tests for JsonFormatter structured logging."""

    def test_json_formatter_outputs_valid_json(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Testing info message",
            args=(),
            exc_info=None,
        )
        record.intended_outcome = "Test that message formats correctly"
        record.actual_outcome = "Message formatted as expected"

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert isinstance(data, dict)
        assert data["message"] == "Testing info message"
        assert data["severity"] == "INFO"
        assert data["logger"] == "test_logger"
        assert data["intended_outcome"] == "Test that message formats correctly"
        assert data["actual_outcome"] == "Message formatted as expected"
        assert "timestamp" in data

    def test_json_formatter_fallback_outcomes_when_omitted(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="fallback_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=20,
            msg="Operation executed",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert "intended_outcome" in data
        assert "actual_outcome" in data
        assert data["intended_outcome"] == "Execute operation: Operation executed"
        assert data["actual_outcome"] == "Completed successfully: Operation executed"

    def test_json_formatter_fallback_error_outcome(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="error_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=30,
            msg="Something broke",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["intended_outcome"] == "Execute operation: Something broke"
        assert data["actual_outcome"] == "Failed with ERROR: Something broke"

    def test_json_formatter_captures_exception_trace(self):
        formatter = JsonFormatter()
        try:
            raise ValueError("Test failure error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="exc_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=40,
            msg="Failed during process",
            args=(),
            exc_info=exc_info,
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert "exception" in data
        assert "Test failure error" in data["exception"]
        assert "Failed with exception: Test failure error" in data["actual_outcome"]

    def test_json_formatter_includes_custom_extras(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="extra_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=50,
            msg="Custom extras message",
            args=(),
            exc_info=None,
        )
        record.intended_outcome = "Log custom data"
        record.actual_outcome = "Data logged"
        record.custom_field = "custom_value_123"
        record.user_id = 42

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["custom_field"] == "custom_value_123"
        assert data["user_id"] == 42


class TestJsonLoggingPlugin:
    """Tests for JsonLoggingPlugin lifecycle callbacks."""

    @pytest.fixture
    def plugin_and_logs(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())

        plugin = JsonLoggingPlugin()
        plugin.logger.addHandler(handler)
        plugin.logger.setLevel(logging.DEBUG)

        yield plugin, stream

        plugin.logger.removeHandler(handler)

    @pytest.mark.asyncio
    async def test_invocation_lifecycle(self, plugin_and_logs):
        plugin, stream = plugin_and_logs

        mock_context = MagicMock()
        mock_context.invocation_id = "inv-123"
        mock_context.agent.name = "test_agent"

        await plugin.before_run_callback(invocation_context=mock_context)
        await plugin.after_run_callback(invocation_context=mock_context)

        lines = [json.loads(line) for line in stream.getvalue().strip().split("\n") if line]
        assert len(lines) == 2

        # Check before_run
        assert lines[0]["event"] == "invocation_start"
        assert "intended_outcome" in lines[0]
        assert "actual_outcome" in lines[0]
        assert lines[0]["invocation_id"] == "inv-123"

        # Check after_run
        assert lines[1]["event"] == "invocation_complete"
        assert "intended_outcome" in lines[1]
        assert "actual_outcome" in lines[1]
        assert lines[1]["invocation_id"] == "inv-123"

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, plugin_and_logs):
        plugin, stream = plugin_and_logs

        mock_agent = MagicMock()
        mock_agent.name = "scheduler_agent"
        mock_agent.description = "Scheduler agent description"
        mock_ctx = MagicMock()
        mock_ctx.agent_name = "scheduler_agent"
        mock_ctx.invocation_id = "inv-456"

        await plugin.before_agent_callback(agent=mock_agent, callback_context=mock_ctx)
        await plugin.after_agent_callback(agent=mock_agent, callback_context=mock_ctx)

        lines = [json.loads(line) for line in stream.getvalue().strip().split("\n") if line]
        assert len(lines) == 2
        assert lines[0]["event"] == "agent_start"
        assert "intended_outcome" in lines[0]
        assert "actual_outcome" in lines[0]
        assert lines[1]["event"] == "agent_complete"
        assert "intended_outcome" in lines[1]
        assert "actual_outcome" in lines[1]

    @pytest.mark.asyncio
    async def test_tool_lifecycle(self, plugin_and_logs):
        plugin, stream = plugin_and_logs

        mock_tool = MagicMock()
        mock_tool.name = "create_event"
        mock_ctx = MagicMock()
        mock_ctx.agent_name = "scheduler_agent"
        mock_ctx.function_call_id = "call-789"
        tool_args = {"summary": "Project Sync", "start_time": "2026-09-10T10:00:00Z"}
        result = {"status": "success", "message": "Meeting scheduled"}

        await plugin.before_tool_callback(tool=mock_tool, tool_args=tool_args, tool_context=mock_ctx)
        await plugin.after_tool_callback(
            tool=mock_tool, tool_args=tool_args, tool_context=mock_ctx, result=result
        )

        lines = [json.loads(line) for line in stream.getvalue().strip().split("\n") if line]
        assert len(lines) == 2
        assert lines[0]["event"] == "tool_start"
        assert lines[0]["tool_name"] == "create_event"
        assert "intended_outcome" in lines[0]
        assert "actual_outcome" in lines[0]

        assert lines[1]["event"] == "tool_complete"
        assert lines[1]["tool_name"] == "create_event"
        assert lines[1]["result_status"] == "success"
        assert "intended_outcome" in lines[1]
        assert "actual_outcome" in lines[1]

    @pytest.mark.asyncio
    async def test_tool_error_callback(self, plugin_and_logs):
        plugin, stream = plugin_and_logs

        mock_tool = MagicMock()
        mock_tool.name = "delete_event"
        mock_ctx = MagicMock()
        mock_ctx.agent_name = "scheduler_agent"
        mock_ctx.function_call_id = "call-err"
        tool_args = {"event_id": "nonexistent"}

        await plugin.on_tool_error_callback(
            tool=mock_tool,
            tool_args=tool_args,
            tool_context=mock_ctx,
            error=RuntimeError("Connection timeout"),
        )

        lines = [json.loads(line) for line in stream.getvalue().strip().split("\n") if line]
        assert len(lines) == 1
        assert lines[0]["event"] == "tool_error"
        assert "Connection timeout" in lines[0]["actual_outcome"]
        assert "intended_outcome" in lines[0]


class TestToolsStructuredLogging:
    """Tests that all tools emit structured JSON logs with intended and actual outcomes."""

    @pytest.fixture
    def tools_log_capture(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())

        tools_logger = logging.getLogger("app.tools")
        original_level = tools_logger.level
        tools_logger.addHandler(handler)
        tools_logger.setLevel(logging.INFO)

        yield stream

        tools_logger.removeHandler(handler)
        tools_logger.setLevel(original_level)

    def test_list_events_logs_outcomes(self, tools_log_capture):
        mock_context = MagicMock()
        mock_context.state = {
            "mock_calendar": True,
            "mock_events": [
                {
                    "id": "mock-event-1",
                    "summary": "Team Standup",
                    "description": "Daily standup",
                    "start": {"dateTime": "2026-09-05T10:00:00Z"},
                    "end": {"dateTime": "2026-09-05T10:30:00Z"},
                }
            ],
        }

        res = list_events(
            start_time="2026-09-05T00:00:00Z",
            end_time="2026-09-05T23:59:59Z",
            query="Standup",
            tool_context=mock_context,
        )
        assert res["status"] == "success"

        logs = [json.loads(line) for line in tools_log_capture.getvalue().strip().split("\n") if line]
        assert len(logs) >= 1
        entry = logs[-1]
        assert "intended_outcome" in entry
        assert "actual_outcome" in entry
        assert "Standup" in entry["intended_outcome"]
        assert "mock calendar" in entry["actual_outcome"]

    def test_check_availability_logs_outcomes(self, tools_log_capture):
        mock_context = MagicMock()
        mock_context.state = {"mock_calendar": True, "mock_events": []}

        res = check_availability(
            start_time="2026-09-05T14:00:00Z",
            end_time="2026-09-05T15:00:00Z",
            tool_context=mock_context,
        )
        assert res["status"] == "success"
        assert res["is_available"] is True

        logs = [json.loads(line) for line in tools_log_capture.getvalue().strip().split("\n") if line]
        assert len(logs) >= 1
        entry = logs[-1]
        assert "intended_outcome" in entry
        assert "actual_outcome" in entry
        assert "available" in entry["actual_outcome"]

    def test_create_event_confirmation_pending_and_success_logs(self, tools_log_capture):
        mock_context = MagicMock()
        mock_context.state = {"mock_calendar": True, "mock_events": []}

        # Step 1: Unconfirmed call
        pending_res = create_event(
            summary="Planning Meeting",
            start_time="2026-09-06T09:00:00Z",
            end_time="2026-09-06T10:00:00Z",
            attendees="alice@example.com",
            description="Sprint planning",
            confirmed=False,
            tool_context=mock_context,
        )
        assert pending_res["status"] == "pending_confirmation"

        # Step 2: Confirmed call
        confirmed_res = create_event(
            summary="Planning Meeting",
            start_time="2026-09-06T09:00:00Z",
            end_time="2026-09-06T10:00:00Z",
            attendees="alice@example.com",
            description="Sprint planning",
            confirmed=True,
            tool_context=mock_context,
        )
        assert confirmed_res["status"] == "success"

        logs = [json.loads(line) for line in tools_log_capture.getvalue().strip().split("\n") if line]
        assert len(logs) >= 2

        pending_log = logs[0]
        assert pending_log["intended_outcome"] == "Schedule calendar event 'Planning Meeting' from 2026-09-06T09:00:00Z to 2026-09-06T10:00:00Z"
        assert "explicit user confirmation" in pending_log["actual_outcome"]

        success_log = logs[1]
        assert "Planning Meeting" in success_log["intended_outcome"]
        assert "Successfully scheduled" in success_log["actual_outcome"]
        assert "[REDACTED_EMAIL]" in success_log["intended_outcome"]
        assert "alice@example.com" not in json.dumps(success_log)

    def test_update_event_logs_outcomes(self, tools_log_capture):
        mock_context = MagicMock()
        mock_context.state = {
            "mock_calendar": True,
            "mock_events": [
                {
                    "id": "event-to-update",
                    "summary": "Original Title",
                    "start": {"dateTime": "2026-09-07T10:00:00Z"},
                    "end": {"dateTime": "2026-09-07T11:00:00Z"},
                }
            ],
        }

        res = update_event(
            event_id="event-to-update",
            summary="Updated Title",
            start_time="",
            end_time="",
            description="",
            confirmed=True,
            tool_context=mock_context,
        )
        assert res["status"] == "success"

        logs = [json.loads(line) for line in tools_log_capture.getvalue().strip().split("\n") if line]
        assert len(logs) >= 1
        entry = logs[-1]
        assert "event-to-update" in entry["intended_outcome"]
        assert "Updated Title" in entry["actual_outcome"]

    def test_delete_event_logs_outcomes(self, tools_log_capture):
        mock_context = MagicMock()
        mock_context.state = {
            "mock_calendar": True,
            "mock_events": [
                {
                    "id": "event-to-delete",
                    "summary": "Meeting to cancel",
                    "start": {"dateTime": "2026-09-08T10:00:00Z"},
                    "end": {"dateTime": "2026-09-08T11:00:00Z"},
                }
            ],
        }

        res = delete_event(
            event_id="event-to-delete",
            confirmed=True,
            tool_context=mock_context,
        )
        assert res["status"] == "success"

        logs = [json.loads(line) for line in tools_log_capture.getvalue().strip().split("\n") if line]
        assert len(logs) >= 1
        entry = logs[-1]
        assert "event-to-delete" in entry["intended_outcome"]
        assert "Successfully deleted" in entry["actual_outcome"]


class TestAgentRoutingLogging:
    """Tests that agent routing emits structured JSON logs."""

    @pytest.fixture
    def agent_log_capture(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())

        agent_logger = logging.getLogger("app.agent")
        original_level = agent_logger.level
        agent_logger.addHandler(handler)
        agent_logger.setLevel(logging.INFO)

        yield stream

        agent_logger.removeHandler(handler)
        agent_logger.setLevel(original_level)

    def test_route_user_request_logs_outcomes(self, agent_log_capture):
        event = route_user_request("I need to schedule focus time for my coding tasks")
        assert event.actions.route == "focus_time"

        logs = [json.loads(line) for line in agent_log_capture.getvalue().strip().split("\n") if line]
        assert len(logs) >= 1
        entry = logs[-1]
        assert entry["event"] == "workflow_routing"
        assert "intended_outcome" in entry
        assert "actual_outcome" in entry
        assert "focus_time" in entry["actual_outcome"]


class TestPiiRedaction:
    """Comprehensive tests for PII redaction across strings, nested structures, and loggers."""

    def test_redact_emails(self):
        sample = "Contact alice@example.com or bob.smith+dev@company.co.uk for scheduling."
        redacted = redact_pii_string(sample)
        assert "[REDACTED_EMAIL]" in redacted
        assert "alice@example.com" not in redacted
        assert "bob.smith+dev@company.co.uk" not in redacted
        assert redacted == "Contact [REDACTED_EMAIL] or [REDACTED_EMAIL] for scheduling."

    def test_redact_phones(self):
        samples = [
            ("Call 555-123-4567 today", "Call [REDACTED_PHONE] today"),
            ("Direct line: (555) 123-4567", "Direct line: [REDACTED_PHONE]"),
            ("Intl: +1 (555) 123-4567", "Intl: [REDACTED_PHONE]"),
            ("Dotted: 555.123.4567", "Dotted: [REDACTED_PHONE]"),
            ("UK office: +44 20 7946 0958", "UK office: [REDACTED_PHONE]"),
            ("France: +33 1 42 68 55 55", "France: [REDACTED_PHONE]"),
        ]
        for original, expected in samples:
            assert redact_pii_string(original) == expected

    def test_phone_redaction_preserves_dates_and_uuids(self):
        text = "Meeting on 2026-09-05 at 10:00:00Z with id mock-event-1 and uuid 6d4a623e-d513-4f4a-a9ba-a30c3a614dd4"
        assert redact_pii_string(text) == text

    def test_redact_social_security_numbers(self):
        sample = "Employee SSN is 000-12-3456 in record"
        redacted = redact_pii_string(sample)
        assert redacted == "Employee SSN is [REDACTED_SSN] in record"

    def test_redact_credit_cards(self):
        samples = [
            ("Card 4111-2222-3333-4444 charged", "Card [REDACTED_CREDIT_CARD] charged"),
            ("Card 4111 2222 3333 4444 charged", "Card [REDACTED_CREDIT_CARD] charged"),
            ("Card 4111222233334444 charged", "Card [REDACTED_CREDIT_CARD] charged"),
        ]
        for orig, exp in samples:
            assert redact_pii_string(orig) == exp

    def test_redact_ip_addresses(self):
        sample = "Client 192.168.1.100 and 10.0.0.5 visited"
        redacted = redact_pii_string(sample)
        assert redacted == "Client [REDACTED_IP] and [REDACTED_IP] visited"

    def test_ip_redaction_preserves_server_bind_addresses(self):
        sample = "Server bound to 0.0.0.0:8000 and 127.0.0.1:8000"
        redacted = redact_pii_string(sample)
        assert redacted == sample

    def test_redact_secrets_and_tokens(self):
        sample = (
            "Auth Header: Bearer ya29.a0AfH6SMA123456789. "
            "Key: AIzaSyD-123456789012345678901234567890. "
            "client_secret: my-secret-key-xyz! "
            "url: https://example.com/api?token=secret123&user=john"
        )
        redacted = redact_pii_string(sample)
        assert "ya29" not in redacted
        assert "[REDACTED_TOKEN]" in redacted
        assert "[REDACTED_API_KEY]" in redacted
        assert "[REDACTED_SECRET]" in redacted
        assert "my-secret-key-xyz!" not in redacted

    def test_redact_nested_structures(self):
        payload = {
            "user_email": "john.doe@example.com",
            "phone_number": "555-987-6543",
            "client_secret": "my-secret-value",
            "access_token": "ya29.xyz",
            "attendees": [
                {"email": "alice@corp.com", "role": "organizer"},
                {"email": "bob@corp.com", "role": "attendee"},
            ],
            "metadata": {
                "user_ip": "172.16.254.1",
                "notes": "Call +1-800-555-0199 about SSN 123-45-6789",
            },
        }

        cleaned = redact_pii(payload)

        assert cleaned["user_email"] == "[REDACTED_EMAIL]"
        assert cleaned["phone_number"] == "[REDACTED_PHONE]"
        assert cleaned["client_secret"] == "[REDACTED_SECRET]"
        assert cleaned["access_token"] == "[REDACTED_SECRET]"
        assert cleaned["attendees"][0]["email"] == "[REDACTED_EMAIL]"
        assert cleaned["attendees"][1]["email"] == "[REDACTED_EMAIL]"
        assert cleaned["metadata"]["user_ip"] == "[REDACTED_IP]"
        assert "[REDACTED_PHONE]" in cleaned["metadata"]["notes"]
        assert "[REDACTED_SSN]" in cleaned["metadata"]["notes"]

    def test_json_formatter_redacts_pii_in_log_records(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="pii_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=50,
            msg="Invited user alice@example.com to meeting, phone 555-234-5678",
            args=(),
            exc_info=None,
        )
        record.intended_outcome = "Schedule meeting with alice@example.com"
        record.actual_outcome = "Sent invite to alice@example.com (555-234-5678)"
        record.attendees = ["alice@example.com", "bob@example.com"]
        record.client_secret = "sensitive-oauth-secret"

        formatted = formatter.format(record)
        data = json.loads(formatted)

        # Check that no raw PII appears anywhere in the serialized JSON
        raw_json = json.dumps(data)
        assert "alice@example.com" not in raw_json
        assert "bob@example.com" not in raw_json
        assert "555-234-5678" not in raw_json
        assert "sensitive-oauth-secret" not in raw_json

        # Verify redacted tokens
        assert "[REDACTED_EMAIL]" in data["message"]
        assert "[REDACTED_PHONE]" in data["message"]
        assert "[REDACTED_EMAIL]" in data["intended_outcome"]
        assert "[REDACTED_EMAIL]" in data["actual_outcome"]
        assert data["attendees"] == ["[REDACTED_EMAIL]", "[REDACTED_EMAIL]"]
        assert data["client_secret"] == "[REDACTED_SECRET]"

    def test_pii_redaction_filter(self):
        filter_ = PiiRedactionFilter()
        record = logging.LogRecord(
            name="filter_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=60,
            msg="User email is %s with pin %s",
            args=("test@example.com", "secret-pin"),
            exc_info=None,
        )
        record.intended_outcome = "Verify test@example.com"
        record.actual_outcome = "Verified test@example.com successfully"

        assert filter_.filter(record) is True
        assert "[REDACTED_EMAIL]" in record.intended_outcome
        assert "[REDACTED_EMAIL]" in record.actual_outcome
        assert "test@example.com" not in record.intended_outcome
        assert "test@example.com" not in record.actual_outcome
