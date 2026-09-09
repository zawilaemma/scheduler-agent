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

"""Unit tests for scheduler-agent tools, validation, and safety constraints."""

from unittest.mock import MagicMock

import pytest
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.tools import AgentTool
from google.adk.workflow import Workflow

from app.agent import (
    CALENDAR_EVENT_ROUTE,
    FOCUS_TIME_ROUTE,
    app,
    is_focus_time_request,
    root_agent,
    route_user_request,
    scheduler_agent,
    task_scheduler_agent,
)
from app.tools import (
    check_availability,
    create_event,
    delete_event,
    list_events,
    negotiate_creds,
    update_event,
)


class DummyToolContext:
    def __init__(self, state=None):
        self.state = state if state is not None else {}
        self.actions = MagicMock()

    def get_auth_response(self, auth_config):
        return None

    def request_credential(self, auth_config):
        pass


def test_agent_structure():
    """Verify root_agent (Workflow), scheduler_agent, and task_scheduler_agent are properly configured."""
    # root_agent is a Workflow agent
    assert isinstance(root_agent, Workflow)
    assert root_agent.name == "root_agent"
    assert app.name == "app"
    assert app.root_agent == root_agent

    # scheduler_agent has the 5 calendar tools
    assert scheduler_agent.name == "scheduler_agent"
    assert len(scheduler_agent.tools) == 5
    tool_names = [getattr(t, "__name__", str(t)) for t in scheduler_agent.tools]
    assert "list_events" in tool_names
    assert "check_availability" in tool_names
    assert "create_event" in tool_names
    assert "update_event" in tool_names
    assert "delete_event" in tool_names

    # task_scheduler_agent has scheduler_agent as an AgentTool
    assert task_scheduler_agent.name == "task_scheduler_agent"
    assert len(task_scheduler_agent.tools) == 1
    assert isinstance(task_scheduler_agent.tools[0], AgentTool)
    assert task_scheduler_agent.tools[0].agent.name == "scheduler_agent"

    # Verify workflow graph edges and routing
    edge_pairs = [(e.from_node.name, e.to_node.name, e.route) for e in root_agent.graph.edges]
    assert ("__START__", "route_user_request", None) in edge_pairs
    assert ("route_user_request", "task_scheduler_agent", FOCUS_TIME_ROUTE) in edge_pairs
    assert ("route_user_request", "scheduler_agent", CALENDAR_EVENT_ROUTE) in edge_pairs


def test_route_focus_time_requests():
    """Verify requests to schedule focus time for tasks route to task_scheduler_agent."""
    focus_queries = [
        "Schedule focus time for writing tests",
        "I need 2 hours of focus time to complete the design doc",
        "Please schedule focus time for a given task",
        "Set aside focus time for bug fixes and code review",
        "Prioritize my tasks and schedule focus time",
        "Can you schedule a focus block tomorrow morning for deep work?",
    ]
    for query in focus_queries:
        assert is_focus_time_request(query) is True
        event = route_user_request(query)
        assert event.actions.route == FOCUS_TIME_ROUTE


def test_route_calendar_event_requests():
    """Verify requests to schedule/list/remove calendar events route directly to scheduler_agent."""
    calendar_queries = [
        "Schedule a meeting with Alex on Friday at 3 PM",
        "Book a 30-minute sync with the product team tomorrow at 10 AM",
        "List all my calendar events for today",
        "Show my upcoming meetings for this week",
        "Remove the team standup on Friday",
        "Delete the event with ID 123",
        "Cancel my 2 PM appointment",
        "Update the meeting at 4 PM to 5 PM",
        "Check my availability on Monday morning",
    ]
    for query in calendar_queries:
        assert is_focus_time_request(query) is False
        event = route_user_request(query)
        assert event.actions.route == CALENDAR_EVENT_ROUTE


def test_route_user_request_content_types():
    """Verify route_user_request handles types.Content and dict inputs properly."""
    from google.genai import types

    content_focus = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Schedule focus time for sprint backlog tasks")],
    )
    event1 = route_user_request(content_focus)
    assert event1.actions.route == FOCUS_TIME_ROUTE
    assert event1.output == content_focus

    content_calendar = types.Content(
        role="user",
        parts=[types.Part.from_text(text="List my calendar events for tomorrow")],
    )
    event2 = route_user_request(content_calendar)
    assert event2.actions.route == CALENDAR_EVENT_ROUTE
    assert event2.output == content_calendar

    dict_focus = {"text": "Schedule focus time for writing documentation"}
    event3 = route_user_request(dict_focus)
    assert event3.actions.route == FOCUS_TIME_ROUTE
    assert event3.output == dict_focus


def test_context_compaction_configured():
    """Verify context compaction is configured on the app."""
    assert app.events_compaction_config is not None
    config = app.events_compaction_config
    assert config.token_threshold == 32000
    assert config.event_retention_size == 5
    assert config.summarizer is not None
    assert isinstance(config.summarizer, LlmEventSummarizer)


@pytest.mark.asyncio
async def test_context_compaction_summarizer():
    """Verify the event summarizer creates a compacted event with model summary."""
    from google.adk.events.event import Event
    from google.genai import types

    mock_llm = MagicMock()
    mock_llm.model = "gemini-3.7-flash"

    async def fake_generate(req, stream=False):
        response = MagicMock()
        response.content = types.Content(
            role="model",
            parts=[types.Part.from_text(text="Compacted summary: user asked for schedule.")],
        )
        response.usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
        )
        yield response

    mock_llm.generate_content_async = fake_generate
    summarizer = LlmEventSummarizer(llm=mock_llm)

    events = [
        Event(
            author="user",
            content=types.Content(role="user", parts=[types.Part.from_text(text="Hello")]),
        ),
        Event(
            author="root_agent",
            content=types.Content(role="model", parts=[types.Part.from_text(text="Hi there")]),
        ),
    ]

    compacted = await summarizer.maybe_summarize_events(events=events)
    assert compacted is not None
    assert compacted.actions.compaction is not None
    assert (
        compacted.actions.compaction.compacted_content.parts[0].text
        == "Compacted summary: user asked for schedule."
    )


def test_safety_guardrail_unconfirmed_create():
    """Verify creating an event without confirmation is rejected with pending_confirmation."""
    ctx = DummyToolContext(state={"mock_calendar": True})
    res = create_event(
        summary="Sprint Planning",
        start_time="2026-09-10T10:00:00Z",
        end_time="2026-09-10T11:00:00Z",
        attendees="team@example.com",
        description="Weekly sprint planning",
        confirmed=False,
        tool_context=ctx,
    )
    assert res["status"] == "pending_confirmation"
    assert "requires user confirmation" in res["message"]
    assert res["event_draft"]["summary"] == "Sprint Planning"


def test_create_and_list_event_flow():
    """Verify event creation with confirmation and subsequent listing."""
    ctx = DummyToolContext(state={"mock_calendar": True})

    # Create with confirmed=True
    create_res = create_event(
        summary="Architecture Review",
        start_time="2026-09-10T14:00:00Z",
        end_time="2026-09-10T15:00:00Z",
        attendees="lead@example.com",
        description="Review system design",
        confirmed=True,
        tool_context=ctx,
    )
    assert create_res["status"] == "success"
    event_id = create_res["event"]["id"]

    # List events
    list_res = list_events(
        start_time="2026-09-10T00:00:00Z",
        end_time="2026-09-10T23:59:59Z",
        query="",
        tool_context=ctx,
    )
    assert list_res["status"] == "success"
    assert list_res["count"] == 1
    assert list_res["events"][0]["summary"] == "Architecture Review"
    assert list_res["events"][0]["id"] == event_id


def test_check_availability_conflicts():
    """Verify check_availability reports conflicts when slots overlap."""
    ctx = DummyToolContext(state={"mock_calendar": True})

    # Book 14:00 - 15:00
    create_event(
        summary="Existing Meeting",
        start_time="2026-09-10T14:00:00Z",
        end_time="2026-09-10T15:00:00Z",
        attendees="",
        description="",
        confirmed=True,
        tool_context=ctx,
    )

    # Check overlapping slot (14:30 - 15:30)
    conflict_check = check_availability(
        start_time="2026-09-10T14:30:00Z",
        end_time="2026-09-10T15:30:00Z",
        tool_context=ctx,
    )
    assert conflict_check["status"] == "success"
    assert conflict_check["is_available"] is False
    assert conflict_check["conflicts_count"] == 1

    # Check free slot (16:00 - 17:00)
    free_check = check_availability(
        start_time="2026-09-10T16:00:00Z",
        end_time="2026-09-10T17:00:00Z",
        tool_context=ctx,
    )
    assert free_check["status"] == "success"
    assert free_check["is_available"] is True
    assert free_check["conflicts_count"] == 0


def test_update_and_delete_event():
    """Verify updating and deleting events with confirmation."""
    ctx = DummyToolContext(state={"mock_calendar": True})

    # Create event
    created = create_event(
        summary="1-on-1 Sync",
        start_time="2026-09-12T09:00:00Z",
        end_time="2026-09-12T09:30:00Z",
        attendees="peer@example.com",
        description="",
        confirmed=True,
        tool_context=ctx,
    )
    event_id = created["event"]["id"]

    # Try update without confirmation
    unconfirmed_update = update_event(
        event_id=event_id,
        summary="Updated 1-on-1 Sync",
        start_time="",
        end_time="",
        description="",
        confirmed=False,
        tool_context=ctx,
    )
    assert unconfirmed_update["status"] == "pending_confirmation"

    # Update with confirmation
    confirmed_update = update_event(
        event_id=event_id,
        summary="Updated 1-on-1 Sync",
        start_time="",
        end_time="",
        description="Updated agenda",
        confirmed=True,
        tool_context=ctx,
    )
    assert confirmed_update["status"] == "success"
    assert confirmed_update["event"]["summary"] == "Updated 1-on-1 Sync"

    # Delete without confirmation
    unconfirmed_del = delete_event(
        event_id=event_id,
        confirmed=False,
        tool_context=ctx,
    )
    assert unconfirmed_del["status"] == "pending_confirmation"

    # Delete with confirmation
    confirmed_del = delete_event(
        event_id=event_id,
        confirmed=True,
        tool_context=ctx,
    )
    assert confirmed_del["status"] == "success"

    # Verify event is removed
    list_res = list_events(
        start_time="2026-09-12T00:00:00Z",
        end_time="2026-09-12T23:59:59Z",
        query="",
        tool_context=ctx,
    )
    assert list_res["count"] == 0


def test_oauth_negotiate_creds_unconfigured():
    """Verify negotiate_creds returns pending without crashing when credentials are not configured."""
    ctx = DummyToolContext(state={})
    ctx.request_credential = MagicMock()
    with pytest.MonkeyPatch.context() as mp:
        from app import auths
        mp.setattr(auths.AUTH_CREDENTIAL.oauth2, "client_id", "")
        mp.setattr(auths.AUTH_CREDENTIAL.oauth2, "client_secret", "")
        res = negotiate_creds(ctx)
        assert isinstance(res, dict)
        assert res.get("pending") is True
        assert "requires OAUTH_CLIENT_ID" in res.get("message", "")
        assert ctx.request_credential.call_count == 0


def test_oauth_negotiate_creds_requests_auth():
    """Verify negotiate_creds calls request_credential when client credentials are set."""
    ctx = DummyToolContext(state={})
    ctx.request_credential = MagicMock()
    with pytest.MonkeyPatch.context() as mp:
        from app import auths
        mp.setattr(auths.AUTH_CREDENTIAL.oauth2, "client_id", "test-client-id")
        mp.setattr(auths.AUTH_CREDENTIAL.oauth2, "client_secret", "test-client-secret")
        res = negotiate_creds(ctx)
        assert isinstance(res, dict)
        assert res.get("pending") is True
        ctx.request_credential.assert_called_once()

