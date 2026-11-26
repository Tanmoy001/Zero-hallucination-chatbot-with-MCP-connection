import base64
import datetime
import functools
import os
import sys
import uuid
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from mcp.server.fastmcp import FastMCP

# This file lives in <project>/mcp_servers/ ; credentials live in <project>/
BASE_DIR = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
TOKEN_PATH = BASE_DIR / "token.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
]

MCP_HOST = "127.0.0.1"
MCP_PORT = int(os.getenv("MCP_GOOGLE_PORT", "8765"))

mcp = FastMCP("GoogleWorkspaceMCP", host=MCP_HOST, port=MCP_PORT)


def _get_credentials(interactive: bool = False) -> Credentials:
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and not creds.valid and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json())
        except RefreshError:
            creds = None  # revoked/expired refresh token -> re-auth needed

    if not creds or not creds.valid:
        if not interactive:
            raise RuntimeError(
                "Google OAuth token missing or expired. Run "
                "`python mcp_servers/google_workspace_mcp.py --auth` first."
            )
        if not CREDENTIALS_PATH.exists():
            raise FileNotFoundError(
                f"Missing {CREDENTIALS_PATH}. Download OAuth Desktop credentials from Google Cloud."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
        creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
    return creds


def _safe(fn):
    """Return errors as text so the LLM can report them instead of crashing the tool call."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except HttpError as exc:
            return f"Google API error: {exc}"
        except Exception as exc:
            return f"Error: {exc}"

    return wrapper


@mcp.tool()
@_safe
def list_recent_emails(max_results: int = 5, query: str = "") -> str:
    """List recent emails from Gmail. Optionally filter using a Gmail search query (e.g. 'from:boss is:unread')."""
    service = build("gmail", "v1", credentials=_get_credentials(), cache_discovery=False)
    res = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    messages = res.get("messages", [])
    if not messages:
        return "No matching emails found."

    output = []
    for m in messages:
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=m["id"], format="metadata", metadataHeaders=["From", "Subject", "Date"])
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        output.append(
            f"- Date: {headers.get('Date', '?')} | From: {headers.get('From', 'Unknown')} | "
            f"Subject: {headers.get('Subject', '(No Subject)')} | Snippet: {msg.get('snippet', '')}"
        )
    return "\n".join(output)


@mcp.tool()
@_safe
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email via Gmail."""
    service = build("gmail", "v1", credentials=_get_credentials(), cache_discovery=False)

    email_msg = EmailMessage()
    email_msg.set_content(body)
    email_msg["To"] = to
    email_msg["Subject"] = subject

    encoded = base64.urlsafe_b64encode(email_msg.as_bytes()).decode()
    sent = service.users().messages().send(userId="me", body={"raw": encoded}).execute()
    return f"Email sent successfully to {to} (Message ID: {sent['id']})."


@mcp.tool()
@_safe
def list_upcoming_meetings(max_results: int = 5) -> str:
    """List upcoming Google Calendar events and Google Meet links."""
    service = build("calendar", "v3", credentials=_get_credentials(), cache_discovery=False)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    events = (
        service.events()
        .list(calendarId="primary", timeMin=now, maxResults=max_results, singleEvents=True, orderBy="startTime")
        .execute()
        .get("items", [])
    )
    if not events:
        return "No upcoming meetings found."

    lines = []
    for ev in events:
        start = ev["start"].get("dateTime", ev["start"].get("date"))
        meet_link = ev.get("hangoutLink", "No Meet link")
        lines.append(f"- {start}: {ev.get('summary', 'Untitled')} | Meet: {meet_link}")
    return "\n".join(lines)


@mcp.tool()
@_safe
def create_google_meet(
    summary: str,
    start_iso: str,
    end_iso: str,
    attendees: Optional[List[str]] = None,
    description: str = "",
) -> str:
    """
    Create a Google Calendar event with an auto-generated Google Meet link.
    start_iso and end_iso must be ISO 8601 strings with timezone offset, e.g. '2026-11-27T15:00:00+01:00'.
    """
    service = build("calendar", "v3", credentials=_get_credentials(), cache_discovery=False)

    event_body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
        "conferenceData": {
            "createRequest": {
                "requestId": str(uuid.uuid4()),
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        },
    }
    if attendees:
        event_body["attendees"] = [{"email": email} for email in attendees]

    created = (
        service.events()
        .insert(calendarId="primary", body=event_body, conferenceDataVersion=1, sendUpdates="all")
        .execute()
    )
    return f"Meeting '{summary}' created. Google Meet Link: {created.get('hangoutLink', 'N/A')}"


if __name__ == "__main__":
    if "--auth" in sys.argv:
        _get_credentials(interactive=True)
        print(f"Authentication successful! Saved token to {TOKEN_PATH}")
    elif "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        # Default: HTTP (works on Windows where psycopg forces the Selector loop,
        # which cannot spawn stdio subprocesses). URL: http://127.0.0.1:8765/mcp
        mcp.run(transport="streamable-http")
