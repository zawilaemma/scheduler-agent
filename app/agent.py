# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events import Event
from google.adk.models import Gemini
from google.adk.tools import AgentTool
from google.adk.workflow import Workflow
from google.genai import types

from app.tools import (
    check_availability,
    create_event,
    delete_event,
    list_events,
    update_event,
)

FLASH_MODEL = "gemini-3.7-flash"
PRO_MODEL = "gemini-3-pro-preview"

SCHEDULER_INSTRUCTION = """You are an intelligent calendar and meeting scheduling assistant.
You help users manage their calendar: checking availability, listing scheduled events, booking meetings, rescheduling, and canceling events.

CAPABILITIES & TOOLS:
- `list_events(start_time, end_time, query)`: Search and view events on the user's primary calendar within a time window (ISO 8601 strings).
- `check_availability(start_time, end_time)`: Check if the user is free or has conflicting commitments during a proposed time slot.
- `create_event(summary, start_time, end_time, attendees, description, confirmed)`: Schedule a new meeting.
- `update_event(event_id, summary, start_time, end_time, description, confirmed)`: Modify an existing meeting's time or details.
- `delete_event(event_id, confirmed)`: Remove or cancel an existing event.

CRITICAL SAFETY & CONFIRMATION RULES:
1. MANDATORY CONFIRMATION BEFORE MODIFICATIONS:
   - You must NEVER schedule, update, or cancel an event without first presenting the proposed details to the user and receiving their explicit confirmation.
   - When a user asks to schedule a meeting:
     a. Check availability for the proposed time using `check_availability` or `list_events`.
     b. Present the drafted event details clearly (Title, Date & Time with timezone, Attendees, and Agenda).
     c. Ask the user: "Would you like me to book this meeting?"
     d. Only after the user confirms (e.g., "Yes", "Please do", "Confirmed"), invoke `create_event` with `confirmed=True`.
   - When a user asks to reschedule or cancel a meeting:
     a. Locate the event with `list_events`.
     b. Present the event details and explicitly ask for confirmation before making changes.
     c. Only after receiving confirmation, call `update_event` or `delete_event` with `confirmed=True`.
   - Never call mutating tools with `confirmed=False` unless requesting validation feedback.

TIMEZONE & DATE HANDLING:
- Always clarify or specify the timezone when proposing or confirming event times.
- If relative dates are mentioned (e.g. "tomorrow", "next Monday"), calculate the proper ISO 8601 UTC or local time string accurately.

AUTHENTICATION:
- If any tool returns a status indicating credentials are pending or authentication is required, politely inform the user that access to their Google Calendar needs to be authorized.
"""

TASK_SCHEDULER_INSTRUCTION = """You are an expert task prioritization and focus time scheduling assistant.
Your goal is to help users organize their work by analyzing tasks, determining priority, estimating necessary focus durations, and scheduling dedicated focus time blocks on their calendar.

ROLES & RESPONSIBILITIES:
1. TASK PRIORITIZATION:
   - Identify all tasks requested by the user.
   - Prioritize them logically based on urgency, importance, effort, and dependencies (e.g. high-impact or quick-win tasks first).
   - Clearly explain the priority ranking to the user.

2. TIME ESTIMATION:
   - Estimate realistic duration for each task (e.g. 30 minutes, 1 hour, 2 hours).
   - If the user provided time constraints or estimates, respect and adapt to them.
   - Clearly present estimated focus durations for each task.

3. FOCUS TIME SCHEDULING:
   - Use the `scheduler_agent` tool to check calendar availability and schedule dedicated focus time blocks for the prioritized tasks.
   - For each task, call `scheduler_agent` with clear details (task title, estimated duration, proposed time window).
   - Present a clear summary of the prioritized plan and the scheduled focus time blocks to the user.
"""


# Define the callback that triggers Memory Bank extraction
async def add_session_to_memory_callback(callback_context: CallbackContext):
    try:
        await callback_context.add_session_to_memory()
    except (ValueError, AttributeError):
        pass
    return None


scheduler_agent = Agent(
    name="scheduler_agent",
    description=(
        "Intelligent calendar and meeting scheduling assistant that manages calendar "
        "events: checking availability, listing scheduled events, booking meetings, "
        "rescheduling, and canceling/deleting events."
    ),
    model=Gemini(
        model=FLASH_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=SCHEDULER_INSTRUCTION,
    tools=[
        list_events,
        check_availability,
        create_event,
        update_event,
        delete_event,
    ],
    # Register the callback so it fires after every interaction
    after_agent_callback=[add_session_to_memory_callback],
)


task_scheduler_agent = Agent(
    name="task_scheduler_agent",
    description=(
        "Task prioritization and focus time scheduling assistant that prioritizes tasks, "
        "estimates how long each task will take, and schedules focus time using scheduler_agent."
    ),
    model=Gemini(
        model=PRO_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=TASK_SCHEDULER_INSTRUCTION,
    tools=[AgentTool(scheduler_agent)],
    after_agent_callback=[add_session_to_memory_callback],
)


def is_focus_time_request(text: str) -> bool:
    """Determine if a user request is for scheduling focus time for tasks."""
    text_lower = text.lower()

    if (
        "focus time" in text_lower
        or "focus block" in text_lower
        or "focus session" in text_lower
        or "deep work" in text_lower
    ):
        return True

    if "focus" in text_lower and any(
        kw in text_lower
        for kw in [
            "task",
            "tasks",
            "work",
            "code",
            "coding",
            "project",
            "writing",
            "study",
            "studying",
            "schedule",
            "block",
            "time",
        ]
    ):
        return True

    if "prioritize" in text_lower and ("task" in text_lower or "schedule" in text_lower):
        return True

    return False


FOCUS_TIME_ROUTE = "focus_time"
CALENDAR_EVENT_ROUTE = "calendar_event"


def route_user_request(node_input: Any) -> Event:
    """Route user request to task_scheduler_agent or scheduler_agent.

    - Focus time requests for tasks route to `task_scheduler_agent`.
    - Direct calendar operations (schedule/list/remove calendar events) route directly to `scheduler_agent`.
    """
    text = ""
    if isinstance(node_input, str):
        text = node_input
    elif hasattr(node_input, "parts"):
        text = "".join(
            part.text for part in node_input.parts if getattr(part, "text", None)
        )
    elif isinstance(node_input, dict):
        if "text" in node_input:
            text = str(node_input["text"])
        elif "message" in node_input:
            msg = node_input["message"]
            if isinstance(msg, str):
                text = msg
            elif isinstance(msg, dict) and "parts" in msg:
                text = "".join(
                    p.get("text", "") for p in msg["parts"] if isinstance(p, dict)
                )
        else:
            text = json.dumps(node_input)
    else:
        text = str(node_input)

    if is_focus_time_request(text):
        return Event(route=FOCUS_TIME_ROUTE, output=node_input)
    return Event(route=CALENDAR_EVENT_ROUTE, output=node_input)


root_agent = Workflow(
    name="root_agent",
    description=(
        "Workflow root agent that inspects incoming user requests and routes them: "
        "focus time task requests to task_scheduler_agent, and direct calendar event "
        "operations directly to scheduler_agent."
    ),
    edges=[
        ("START", route_user_request),
        (
            route_user_request,
            {
                FOCUS_TIME_ROUTE: task_scheduler_agent,
                CALENDAR_EVENT_ROUTE: scheduler_agent,
            },
        ),
    ],
)


app = App(
    root_agent=root_agent,
    name="app",
    events_compaction_config=EventsCompactionConfig(
        token_threshold=32000,
        event_retention_size=5,
        summarizer=LlmEventSummarizer(
            llm=Gemini(
                model=MODEL,
                retry_options=types.HttpRetryOptions(attempts=3),
            )
        ),
    ),
)
