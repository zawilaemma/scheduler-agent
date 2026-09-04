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

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini
from google.genai import types

from app.tools import (
    check_availability,
    create_event,
    delete_event,
    list_events,
    update_event,
)

from google.adk.agents.callback_context import CallbackContext

MODEL = "gemini-3.7-flash"


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

# Define the callback that triggers Memory Bank extraction
async def add_session_to_memory_callback(callback_context: CallbackContext):
    await callback_context.add_session_to_memory()
    return None


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model=MODEL,
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

