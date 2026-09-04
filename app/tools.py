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

"""Tools for the Google Calendar scheduler agent.

Implements the OAuth 2.0 credential negotiation pattern from the
oauth-user-consent-flow reference recipe, supporting both production
Google Calendar API and a local mock mode for testing and evaluation.
"""

import json
import logging
import os
import uuid
from typing import Any

from google.adk.tools import ToolContext
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from app import auths

logger = logging.getLogger(__name__)


def negotiate_creds(tool_context: ToolContext) -> Credentials | dict:
    """Handle the OAuth 2.0 flow to get valid credentials.

    Three-stage credential resolution:
    1. Check for cached/injected token in tool_context.state.
    2. Check for auth response from the ADK OAuth flow.
    3. If nothing is available, request credentials from user.
    """
    logger.info("Negotiating credentials using OAuth 2.0")

    # --- Stage 1: Check for cached / injected token ---
    cached_token = tool_context.state.get(auths.TOKEN_CACHE_KEY)
    if cached_token is None:
        cached_token = tool_context.state.get(f"temp:{auths.TOKEN_CACHE_KEY}")

    if cached_token:
        logger.debug("Found cached token in tool context state")
        if isinstance(cached_token, dict):
            try:
                creds = Credentials.from_authorized_user_info(
                    cached_token, list(auths.SCOPES.keys())
                )
                if creds.valid:
                    return creds
                if creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    tool_context.state[auths.TOKEN_CACHE_KEY] = json.loads(creds.to_json())
                    return creds
            except Exception as error:
                logger.error(f"Error refreshing credentials: {error}")
                tool_context.state[auths.TOKEN_CACHE_KEY] = None

        elif isinstance(cached_token, str):
            return Credentials(token=cached_token)
        else:
            raise ValueError(f"Invalid cached token type: {type(cached_token)}")

    # --- Stage 2: Check for auth response from ADK OAuth flow ---
    if exchanged_creds := tool_context.get_auth_response(auths.AUTH_CONFIG):
        auth_scheme = auths.AUTH_CONFIG.auth_scheme
        auth_credential = auths.AUTH_CONFIG.raw_auth_credential
        creds = Credentials(
            token=exchanged_creds.oauth2.access_token,
            refresh_token=exchanged_creds.oauth2.refresh_token,
            token_uri=auth_scheme.flows.authorizationCode.tokenUrl,
            client_id=auth_credential.oauth2.client_id,
            client_secret=auth_credential.oauth2.client_secret,
            scopes=list(auth_scheme.flows.authorizationCode.scopes.keys()),
        )
        tool_context.state[auths.TOKEN_CACHE_KEY] = json.loads(creds.to_json())
        return creds

    # --- Stage 3: Initiate OAuth flow ---
    client_id = getattr(auths.AUTH_CREDENTIAL.oauth2, "client_id", "")
    client_secret = getattr(auths.AUTH_CREDENTIAL.oauth2, "client_secret", "")
    if not (client_id and client_secret):
        logger.info("OAuth client credentials not configured; awaiting setup.")
        return {
            "pending": True,
            "message": "Google Calendar authentication requires OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET in the environment.",
        }

    tool_context.request_credential(auths.AUTH_CONFIG)
    return {"pending": True, "message": "Awaiting user authentication with Google Calendar"}


def _get_backend(tool_context: ToolContext) -> tuple[Any, Any]:
    """Resolves whether to use Google Calendar API service or local mock store.

    Returns:
        (service, None) if Google Calendar API client is available.
        (None, mock_events_list) if mock mode is active.
        (None, dict) if credentials negotiation is pending.
    """
    is_mock = tool_context.state.get("mock_calendar") or os.environ.get("MOCK_CALENDAR", "").lower() in ("true", "1")
    if is_mock:
        if "mock_events" not in tool_context.state:
            tool_context.state["mock_events"] = [
                {
                    "id": "mock-event-1",
                    "summary": "Team Standup",
                    "description": "Daily standup",
                    "start": {"dateTime": "2026-09-05T10:00:00Z"},
                    "end": {"dateTime": "2026-09-05T10:30:00Z"},
                    "attendees": ["team@example.com"],
                    "htmlLink": "https://calendar.google.com/calendar/event?eid=mock-event-1",
                }
            ]
        return None, tool_context.state["mock_events"]

    creds = negotiate_creds(tool_context)
    if isinstance(creds, dict):
        return None, creds

    service = build("calendar", "v3", credentials=creds)
    return service, None


def list_events(start_time: str, end_time: str, query: str, tool_context: ToolContext) -> dict:
    """Lists calendar events within the given time range.

    Args:
        start_time: ISO 8601 start time (e.g. '2026-09-05T00:00:00Z').
        end_time: ISO 8601 end time (e.g. '2026-09-05T23:59:59Z').
        query: Optional search keyword to filter events, or empty string.

    Returns:
        dict with status and list of matching calendar events.
    """
    service, backend = _get_backend(tool_context)
    if service is None and isinstance(backend, dict) and backend.get("pending"):
        return backend

    if service is not None:
        try:
            params: dict[str, Any] = {
                "calendarId": "primary",
                "timeMin": start_time,
                "timeMax": end_time,
                "singleEvents": True,
                "orderBy": "startTime",
            }
            if query:
                params["q"] = query

            events_result = service.events().list(**params).execute()
            items = events_result.get("items", [])
            formatted = []
            for item in items:
                formatted.append({
                    "id": item.get("id"),
                    "summary": item.get("summary", "No Title"),
                    "start": item.get("start", {}).get("dateTime") or item.get("start", {}).get("date"),
                    "end": item.get("end", {}).get("dateTime") or item.get("end", {}).get("date"),
                    "description": item.get("description", ""),
                    "attendees": [att.get("email") for att in item.get("attendees", []) if "email" in att],
                    "htmlLink": item.get("htmlLink", ""),
                })
            return {"status": "success", "count": len(formatted), "events": formatted}
        except Exception as e:
            logger.error(f"Error querying Google Calendar API: {e}")
            return {"status": "error", "message": f"Failed to list events: {e!s}"}

    # Local mock mode
    mock_events = backend
    results = []
    q_lower = query.lower() if query else ""
    for ev in mock_events:
        ev_start = ev.get("start", {}).get("dateTime", "")
        ev_end = ev.get("end", {}).get("dateTime", "")
        # Filter within range if provided
        if ev_end and start_time and ev_end < start_time:
            continue
        if ev_start and end_time and ev_start > end_time:
            continue
        if q_lower:
            summary = ev.get("summary", "").lower()
            desc = ev.get("description", "").lower()
            if q_lower not in summary and q_lower not in desc:
                continue
        results.append(ev)

    return {"status": "success", "count": len(results), "events": results}


def check_availability(start_time: str, end_time: str, tool_context: ToolContext) -> dict:
    """Checks whether the user has free time or conflicts during a specific time range.

    Args:
        start_time: ISO 8601 start time of the proposed meeting.
        end_time: ISO 8601 end time of the proposed meeting.

    Returns:
        dict with status, availability (True/False), and conflicting events if any.
    """
    events_res = list_events(start_time=start_time, end_time=end_time, query="", tool_context=tool_context)
    if events_res.get("status") != "success":
        return events_res

    conflicts = events_res.get("events", [])
    is_available = len(conflicts) == 0

    return {
        "status": "success",
        "is_available": is_available,
        "conflicts_count": len(conflicts),
        "conflicts": conflicts,
    }


def create_event(
    summary: str,
    start_time: str,
    end_time: str,
    attendees: str,
    description: str,
    confirmed: bool,
    tool_context: ToolContext,
) -> dict:
    """Creates a new calendar meeting after user confirmation.

    Args:
        summary: Title of the meeting.
        start_time: ISO 8601 start time (e.g. '2026-09-05T14:00:00Z').
        end_time: ISO 8601 end time (e.g. '2026-09-05T14:30:00Z').
        attendees: Comma-separated email addresses of attendees, or empty string.
        description: Description or agenda of the meeting, or empty string.
        confirmed: Boolean indicating whether the user has explicitly confirmed booking this event. Must be True to proceed.

    Returns:
        dict with status, event details, and confirmation message.
    """
    if not confirmed:
        return {
            "status": "pending_confirmation",
            "message": (
                f"Action requires user confirmation. Please present the event details to the user: "
                f"Title: '{summary}', Time: {start_time} to {end_time}, Attendees: {attendees or 'None'}, "
                f"Description: {description or 'None'}. Once the user explicitly agrees, call create_event with confirmed=True."
            ),
            "event_draft": {
                "summary": summary,
                "start_time": start_time,
                "end_time": end_time,
                "attendees": attendees,
                "description": description,
            },
        }

    service, backend = _get_backend(tool_context)
    if service is None and isinstance(backend, dict) and backend.get("pending"):
        return backend

    parsed_attendees = [
        {"email": email.strip()}
        for email in attendees.split(",")
        if email.strip()
    ]

    if service is not None:
        try:
            body: dict[str, Any] = {
                "summary": summary,
                "description": description,
                "start": {"dateTime": start_time},
                "end": {"dateTime": end_time},
            }
            if parsed_attendees:
                body["attendees"] = parsed_attendees

            created = service.events().insert(calendarId="primary", body=body).execute()
            return {
                "status": "success",
                "message": f"Successfully scheduled meeting '{summary}'.",
                "event": {
                    "id": created.get("id"),
                    "summary": created.get("summary"),
                    "start": created.get("start", {}).get("dateTime"),
                    "end": created.get("end", {}).get("dateTime"),
                    "htmlLink": created.get("htmlLink", ""),
                },
            }
        except Exception as e:
            logger.error(f"Error creating event in Google Calendar: {e}")
            return {"status": "error", "message": f"Failed to create event: {e!s}"}

    # Local mock mode
    mock_events = backend
    event_id = str(uuid.uuid4())[:8]
    new_event = {
        "id": event_id,
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_time},
        "end": {"dateTime": end_time},
        "attendees": [att["email"] for att in parsed_attendees],
        "htmlLink": f"https://calendar.google.com/calendar/event?eid={event_id}",
    }
    mock_events.append(new_event)
    return {
        "status": "success",
        "message": f"Successfully scheduled meeting '{summary}'.",
        "event": new_event,
    }


def update_event(
    event_id: str,
    summary: str,
    start_time: str,
    end_time: str,
    description: str,
    confirmed: bool,
    tool_context: ToolContext,
) -> dict:
    """Updates an existing calendar meeting after user confirmation.

    Args:
        event_id: The ID of the event to update.
        summary: New title for the meeting, or empty string to keep unchanged.
        start_time: New ISO 8601 start time, or empty string to keep unchanged.
        end_time: New ISO 8601 end time, or empty string to keep unchanged.
        description: New description for the meeting, or empty string to keep unchanged.
        confirmed: Boolean indicating whether the user has explicitly confirmed updating this event. Must be True to proceed.

    Returns:
        dict with status and updated event details.
    """
    if not confirmed:
        return {
            "status": "pending_confirmation",
            "message": (
                f"Action requires user confirmation. Please ask the user to confirm updating event ID '{event_id}'. "
                f"Once confirmed by the user, call update_event with confirmed=True."
            ),
            "event_draft": {
                "event_id": event_id,
                "summary": summary,
                "start_time": start_time,
                "end_time": end_time,
                "description": description,
            },
        }

    service, backend = _get_backend(tool_context)
    if service is None and isinstance(backend, dict) and backend.get("pending"):
        return backend

    if service is not None:
        try:
            body: dict[str, Any] = {}
            if summary:
                body["summary"] = summary
            if description:
                body["description"] = description
            if start_time:
                body["start"] = {"dateTime": start_time}
            if end_time:
                body["end"] = {"dateTime": end_time}

            updated = service.events().patch(calendarId="primary", eventId=event_id, body=body).execute()
            return {
                "status": "success",
                "message": f"Successfully updated meeting '{updated.get('summary')}'.",
                "event": updated,
            }
        except Exception as e:
            logger.error(f"Error updating event in Google Calendar: {e}")
            return {"status": "error", "message": f"Failed to update event: {e!s}"}

    # Local mock mode
    mock_events = backend
    for ev in mock_events:
        if ev.get("id") == event_id:
            if summary:
                ev["summary"] = summary
            if description:
                ev["description"] = description
            if start_time:
                ev["start"] = {"dateTime": start_time}
            if end_time:
                ev["end"] = {"dateTime": end_time}
            return {
                "status": "success",
                "message": f"Successfully updated meeting '{ev.get('summary')}'.",
                "event": ev,
            }

    return {"status": "error", "message": f"Event with ID '{event_id}' not found."}


def delete_event(
    event_id: str,
    confirmed: bool,
    tool_context: ToolContext,
) -> dict:
    """Deletes or cancels an existing calendar meeting after user confirmation.

    Args:
        event_id: The ID of the event to delete.
        confirmed: Boolean indicating whether the user has explicitly confirmed deleting this event. Must be True to proceed.

    Returns:
        dict with status and cancellation message.
    """
    if not confirmed:
        return {
            "status": "pending_confirmation",
            "message": (
                f"Action requires user confirmation. Please ask the user to confirm deleting event ID '{event_id}'. "
                f"Once confirmed by the user, call delete_event with confirmed=True."
            ),
        }

    service, backend = _get_backend(tool_context)
    if service is None and isinstance(backend, dict) and backend.get("pending"):
        return backend

    if service is not None:
        try:
            service.events().delete(calendarId="primary", eventId=event_id).execute()
            return {
                "status": "success",
                "message": f"Successfully deleted event with ID '{event_id}'.",
            }
        except Exception as e:
            logger.error(f"Error deleting event in Google Calendar: {e}")
            return {"status": "error", "message": f"Failed to delete event: {e!s}"}

    # Local mock mode
    mock_events = backend
    for i, ev in enumerate(mock_events):
        if ev.get("id") == event_id:
            removed = mock_events.pop(i)
            return {
                "status": "success",
                "message": f"Successfully deleted event '{removed.get('summary')}' (ID: {event_id}).",
            }

    return {"status": "error", "message": f"Event with ID '{event_id}' not found."}
