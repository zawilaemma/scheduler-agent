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

"""Structured JSON logging configuration for the scheduler agent.

Configures application-wide JSON logging so all log entries are formatted as
single-line JSON objects, ensuring that the agent's intended outcome and actual
outcome are systematically recorded in every log entry.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import sys
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InvocationContext
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

RESERVED_LOG_RECORD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}

SENSITIVE_KEY_NAMES = {
    "password",
    "secret",
    "client_secret",
    "access_token",
    "refresh_token",
    "auth_token",
    "private_key",
    "api_key",
    "apikey",
    "credential",
}

# Regex patterns for Personally Identifiable Information (PII)
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(
    r"(?<!\w)"
    r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*\d{3}\s*\)|\b\d{3}\b)\s*(?:[.-]\s*)?\d{3}\s*(?:[.-]\s*)?\d{4}\b"
    r"|"
    r"\+\d{1,3}(?:[ .-]?\d{1,4}){2,5}\b"
    r")"
)
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
IP_REGEX = re.compile(
    r"\b(?!(?:0\.0\.0\.0|127\.0\.0\.1)\b)(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)
BEARER_REGEX = re.compile(r"(?i)\b(bearer\s+)[a-zA-Z0-9_\-\.]+")
YA29_REGEX = re.compile(r"\bya29\.[a-zA-Z0-9_\-]+")
API_KEY_REGEX = re.compile(r"\bAIza[0-9A-Za-z-_]{30,40}\b")
KV_SECRET_REGEX = re.compile(
    r"(?i)\b(client_secret|password|passwd|secret|api_key|apikey|access_token|refresh_token|private_key)\s*[:=]\s*['\"]?([^\s'\",;]+)['\"]?"
)
URL_SECRET_REGEX = re.compile(
    r"(?i)([?&](?:token|key|api_key|secret|password|access_token)=)[^&\s]+"
)


def redact_pii_string(text: str) -> str:
    """Scrub PII (email, phone, SSN, credit cards, IP addresses, secrets/tokens) from a string."""
    if not isinstance(text, str):
        return text
    text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    text = SSN_REGEX.sub("[REDACTED_SSN]", text)
    text = CREDIT_CARD_REGEX.sub("[REDACTED_CREDIT_CARD]", text)
    text = BEARER_REGEX.sub(r"\g<1>[REDACTED_TOKEN]", text)
    text = YA29_REGEX.sub("[REDACTED_TOKEN]", text)
    text = API_KEY_REGEX.sub("[REDACTED_API_KEY]", text)
    text = KV_SECRET_REGEX.sub(r"\g<1>: [REDACTED_SECRET]", text)
    text = URL_SECRET_REGEX.sub(r"\g<1>[REDACTED_SECRET]", text)
    text = IP_REGEX.sub("[REDACTED_IP]", text)
    text = PHONE_REGEX.sub("[REDACTED_PHONE]", text)
    return text


def redact_pii(data: Any) -> Any:
    """Recursively scrub PII and sensitive keys from arbitrary data structures."""
    if isinstance(data, str):
        return redact_pii_string(data)
    elif isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(k, str) and any(
                secret_word in k.lower() for secret_word in SENSITIVE_KEY_NAMES
            ):
                cleaned[k] = "[REDACTED_SECRET]"
            else:
                cleaned[k] = redact_pii(v)
        return cleaned
    elif isinstance(data, list):
        return [redact_pii(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(redact_pii(item) for item in data)
    elif isinstance(data, set):
        return {redact_pii(item) for item in data}
    return data


class PiiRedactionFilter(logging.Filter):
    """Logging filter that redacts PII in LogRecord messages, arguments, and outcomes."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_pii_string(record.msg)
        elif isinstance(record.msg, (dict, list)):
            record.msg = redact_pii(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_pii(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(redact_pii(arg) for arg in record.args)

        if hasattr(record, "intended_outcome") and isinstance(record.intended_outcome, str):
            record.intended_outcome = redact_pii_string(record.intended_outcome)
        if hasattr(record, "actual_outcome") and isinstance(record.actual_outcome, str):
            record.actual_outcome = redact_pii_string(record.actual_outcome)

        return True


class JsonFormatter(logging.Formatter):
    """Logging formatter that serializes log records into structured JSON lines.

    Guarantees that every log record contains 'intended_outcome' and 'actual_outcome',
    recording the agent's intentions alongside observed outcomes.
    Also redacts any PII (emails, phone numbers, SSNs, credit cards, IP addresses, secrets)
    from all log records before output.
    """

    def __init__(self, *args: Any, redact: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.redact = redact

    def format(self, record: logging.LogRecord) -> str:
        raw_msg = record.msg
        message_str = record.getMessage()

        intended_outcome = getattr(record, "intended_outcome", None)
        actual_outcome = getattr(record, "actual_outcome", None)

        log_entry: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "severity": record.levelname,
            "level": record.levelname,
            "logger": record.name,
            "message": message_str,
            "intended_outcome": intended_outcome,
            "actual_outcome": actual_outcome,
        }

        # Include custom extra fields
        for key, val in record.__dict__.items():
            if key not in RESERVED_LOG_RECORD_ATTRS and key not in log_entry:
                log_entry[key] = val

        # If msg itself was a dictionary, overlay its keys
        if isinstance(raw_msg, dict):
            for k, v in raw_msg.items():
                log_entry[k] = v

        # Fallback if intended_outcome is still None
        if log_entry.get("intended_outcome") is None:
            log_entry["intended_outcome"] = f"Execute operation: {log_entry.get('message', '')}"

        # Fallback if actual_outcome is still None
        if log_entry.get("actual_outcome") is None:
            if record.exc_info and record.exc_info[1]:
                log_entry["actual_outcome"] = f"Failed with exception: {record.exc_info[1]}"
            elif record.levelno >= logging.ERROR:
                log_entry["actual_outcome"] = f"Failed with {record.levelname}: {log_entry.get('message', '')}"
            elif record.levelno >= logging.WARNING:
                log_entry["actual_outcome"] = f"Completed with warning: {log_entry.get('message', '')}"
            else:
                log_entry["actual_outcome"] = f"Completed successfully: {log_entry.get('message', '')}"

        # Include exception trace if present
        if record.exc_info and "exception" not in log_entry:
            log_entry["exception"] = self.formatException(record.exc_info)
        if record.stack_info and "stack_info" not in log_entry:
            log_entry["stack_info"] = self.formatStack(record.stack_info)

        # Redact PII across the entire log entry before serializing
        if self.redact:
            log_entry = redact_pii(log_entry)

        return json.dumps(log_entry, default=str)


_logging_initialized = False


def setup_logging(level: int | str = logging.INFO, *, force: bool = False) -> None:
    """Configures structured JSON logging globally across the application with PII redaction."""
    global _logging_initialized
    if _logging_initialized and not force:
        return

    root_logger = logging.getLogger()
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid plain text logging duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    pii_filter = PiiRedactionFilter()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JsonFormatter(redact=True))
    stream_handler.addFilter(pii_filter)
    stream_handler.setLevel(level)
    root_logger.addHandler(stream_handler)
    root_logger.addFilter(pii_filter)

    # Also configure uvicorn loggers if they exist
    for uvicorn_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(uvicorn_name)
        uv_logger.handlers = [stream_handler]
        uv_logger.addFilter(pii_filter)
        uv_logger.propagate = False

    _logging_initialized = True


class JsonLoggingPlugin(BasePlugin):
    """ADK runner plugin that logs lifecycle events as structured JSON lines.

    Records the agent's intended outcome as well as actual outcome at each stage
    of execution: invocations, agent steps, tool calls, and model requests.
    """

    def __init__(self, name: str = "json_logging_plugin"):
        super().__init__(name=name)
        self.logger = logging.getLogger("app.plugin.json_logging")

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        agent_name = (
            invocation_context.agent.name
            if invocation_context.agent and hasattr(invocation_context.agent, "name")
            else "Unknown"
        )
        self.logger.info(
            f"Invocation starting for agent '{agent_name}'",
            extra={
                "event": "invocation_start",
                "invocation_id": invocation_context.invocation_id,
                "agent_name": agent_name,
                "intended_outcome": f"Process user request and execute workflow for agent '{agent_name}'",
                "actual_outcome": "Invocation initiated",
            },
        )
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        agent_name = (
            invocation_context.agent.name
            if invocation_context.agent and hasattr(invocation_context.agent, "name")
            else "Unknown"
        )
        self.logger.info(
            f"Invocation completed for agent '{agent_name}'",
            extra={
                "event": "invocation_complete",
                "invocation_id": invocation_context.invocation_id,
                "agent_name": agent_name,
                "intended_outcome": f"Successfully complete user request workflow for agent '{agent_name}'",
                "actual_outcome": "Invocation completed successfully",
            },
        )
        return None

    async def on_run_error_callback(
        self, *, invocation_context: InvocationContext, error: Exception
    ) -> None:
        agent_name = (
            invocation_context.agent.name
            if invocation_context.agent and hasattr(invocation_context.agent, "name")
            else "Unknown"
        )
        self.logger.error(
            f"Invocation failed for agent '{agent_name}': {error}",
            extra={
                "event": "invocation_error",
                "invocation_id": invocation_context.invocation_id,
                "agent_name": agent_name,
                "intended_outcome": f"Execute invocation for agent '{agent_name}'",
                "actual_outcome": f"Invocation failed with error: {error}",
            },
        )
        return None

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        agent_name = getattr(agent, "name", callback_context.agent_name)
        description = getattr(agent, "description", "")
        self.logger.info(
            f"Agent '{agent_name}' starting execution",
            extra={
                "event": "agent_start",
                "agent_name": agent_name,
                "invocation_id": callback_context.invocation_id,
                "intended_outcome": f"Execute agent '{agent_name}': {description or 'manage user tasks'}",
                "actual_outcome": f"Agent '{agent_name}' started execution",
            },
        )
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        agent_name = getattr(agent, "name", callback_context.agent_name)
        self.logger.info(
            f"Agent '{agent_name}' completed execution",
            extra={
                "event": "agent_complete",
                "agent_name": agent_name,
                "invocation_id": callback_context.invocation_id,
                "intended_outcome": f"Complete agent '{agent_name}' execution and handle user request",
                "actual_outcome": f"Agent '{agent_name}' completed execution",
            },
        )
        return None

    async def on_agent_error_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext, error: Exception
    ) -> None:
        agent_name = getattr(agent, "name", callback_context.agent_name)
        self.logger.error(
            f"Agent '{agent_name}' encountered error: {error}",
            extra={
                "event": "agent_error",
                "agent_name": agent_name,
                "invocation_id": callback_context.invocation_id,
                "intended_outcome": f"Execute agent '{agent_name}' successfully",
                "actual_outcome": f"Agent execution failed with error: {error}",
            },
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        self.logger.info(
            f"Tool '{tool.name}' starting execution",
            extra={
                "event": "tool_start",
                "tool_name": tool.name,
                "agent_name": tool_context.agent_name,
                "function_call_id": tool_context.function_call_id,
                "tool_args": tool_args,
                "intended_outcome": f"Execute tool '{tool.name}' with arguments {tool_args}",
                "actual_outcome": f"Tool '{tool.name}' execution initiated",
            },
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        status = result.get("status", "success") if isinstance(result, dict) else "success"
        message = result.get("message", "") if isinstance(result, dict) else ""
        self.logger.info(
            f"Tool '{tool.name}' completed execution",
            extra={
                "event": "tool_complete",
                "tool_name": tool.name,
                "agent_name": tool_context.agent_name,
                "function_call_id": tool_context.function_call_id,
                "tool_args": tool_args,
                "result_status": status,
                "intended_outcome": f"Execute tool '{tool.name}' to perform calendar operation",
                "actual_outcome": f"Tool '{tool.name}' completed with status '{status}'{f': {message}' if message else ''}",
            },
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        self.logger.error(
            f"Tool '{tool.name}' failed: {error}",
            extra={
                "event": "tool_error",
                "tool_name": tool.name,
                "agent_name": tool_context.agent_name,
                "function_call_id": tool_context.function_call_id,
                "tool_args": tool_args,
                "intended_outcome": f"Execute tool '{tool.name}' successfully",
                "actual_outcome": f"Tool '{tool.name}' execution failed with error: {error}",
            },
        )
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        model_name = llm_request.model or "default"
        self.logger.info(
            f"Sending LLM request to model '{model_name}'",
            extra={
                "event": "model_request",
                "model": model_name,
                "agent_name": callback_context.agent_name,
                "intended_outcome": f"Generate LLM response from model '{model_name}' for agent '{callback_context.agent_name}'",
                "actual_outcome": "LLM request dispatched to model",
            },
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if llm_response.error_code:
            self.logger.error(
                f"LLM error: {llm_response.error_message}",
                extra={
                    "event": "model_response_error",
                    "agent_name": callback_context.agent_name,
                    "error_code": llm_response.error_code,
                    "intended_outcome": f"Receive valid model generation for agent '{callback_context.agent_name}'",
                    "actual_outcome": f"Model returned error {llm_response.error_code}: {llm_response.error_message}",
                },
            )
        else:
            self.logger.info(
                f"Received LLM response for agent '{callback_context.agent_name}'",
                extra={
                    "event": "model_response",
                    "agent_name": callback_context.agent_name,
                    "intended_outcome": f"Receive model generation for agent '{callback_context.agent_name}'",
                    "actual_outcome": "Successfully received LLM response",
                },
            )
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> LlmResponse | None:
        self.logger.error(
            f"LLM model error: {error}",
            extra={
                "event": "model_error",
                "agent_name": callback_context.agent_name,
                "intended_outcome": f"Generate content from model for agent '{callback_context.agent_name}'",
                "actual_outcome": f"LLM call failed with error: {error}",
            },
        )
        return None

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Event | None:
        self.logger.info(
            f"Event yielded: {event.id} by {event.author}",
            extra={
                "event": "event_yielded",
                "event_id": event.id,
                "author": event.author,
                "is_final": event.is_final_response(),
                "intended_outcome": f"Stream event '{event.id}' from '{event.author}' to caller",
                "actual_outcome": f"Event yielded successfully (final_response={event.is_final_response()})",
            },
        )
        return None


__all__ = [
    "CREDIT_CARD_REGEX",
    "EMAIL_REGEX",
    "IP_REGEX",
    "PHONE_REGEX",
    "SSN_REGEX",
    "JsonFormatter",
    "JsonLoggingPlugin",
    "PiiRedactionFilter",
    "redact_pii",
    "redact_pii_string",
    "setup_logging",
]
