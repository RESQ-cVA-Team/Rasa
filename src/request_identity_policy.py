"""Which identity, if any, a request to the Rasa HTTP server must prove.

Pure functions only (no Sanic, no environment), so the policy can be tested
on its own. run_rasa.py applies it to every request as the single
authentication gate: Rasa itself is started without a static auth token, so
anything this module does not explicitly allow is denied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

WEBHOOK_PATH = "/webhooks/rest/webhook"

# Keycloak realm role a token must carry to use cVA at all -- mirrors
# Webapp's REQUIRED_CVA_ROLE (src/lib/cvaAccess.ts) and Action's own check.
REQUIRED_ROLE = "cva"

_SENDER_THREAD_SUFFIX_RE = re.compile(r"^(.*):thread:(\d+)$")

# Rasa's built-in conversation routes plus the custom DELETE .../tracker route.
# Anchored at both ends, so any extra path segments end up inside the sender id
# and fail the identity match instead of slipping past it.
_CONVERSATION_RE = re.compile(
    r"^/conversations/(?P<sender>.+)/"
    r"(?:tracker(?:/events)?|story|execute|trigger_intent|predict|messages)$"
)
_THREADS_BY_USER_RE = re.compile(r"^/threads/by-user/(?P<sub>[^/]+)(?:/next-id)?$")
_THREADS_INDEX_EVENT_RE = re.compile(r"^/threads/(?P<sub>[^/]+)/index-event$")
_THREADS_DELETE_RE = re.compile(r"^/threads/(?P<sub>[^/]+)/thread/[^/]+$")


def sender_sub(sender_id: str) -> str:
    """Strip the `:thread:<id>` suffix, mirroring rasaSender.ts's parseRasaSenderId."""
    match = _SENDER_THREAD_SUFFIX_RE.match(sender_id)
    return match.group(1) if match else sender_id


def has_required_role(introspection_payload: dict) -> bool:
    """Realm roles only, matching Webapp's cvaAccess.ts: Keycloak's introspect
    response mirrors the token's own claims for an active token, so this reads
    the top-level `roles` claim and `realm_access.roles` -- never groups or
    client/resource roles, so a group or client role named "cva" doesn't count."""
    roles: list[str] = []
    for value in (introspection_payload.get("roles"), (introspection_payload.get("realm_access") or {}).get("roles")):
        if isinstance(value, list):
            roles.extend(entry.strip().lower() for entry in value if isinstance(entry, str) and entry.strip())
    return REQUIRED_ROLE in roles


@dataclass(frozen=True)
class Decision:
    """`open`: no credentials needed. `deny`: never allowed. `malformed`: the
    request cannot be bound to a user (webhook without a sender). `bound`: a
    verified user token whose subject equals `sub` is required."""

    kind: str
    sub: Optional[str] = None


OPEN = Decision("open")
DENY = Decision("deny")
MALFORMED = Decision("malformed")


def required_identity(method: str, path: str, body_sender: Optional[str]) -> Decision:
    method = method.upper()

    # CORS preflight carries no credentials by design.
    if method == "OPTIONS":
        return OPEN

    if path == "/version":
        return OPEN if method in {"GET", "HEAD"} else DENY

    if path == WEBHOOK_PATH:
        if method != "POST":
            return DENY
        if not body_sender:
            return MALFORMED
        return Decision("bound", sender_sub(body_sender))

    conversation = _CONVERSATION_RE.match(path)
    if conversation:
        return Decision("bound", sender_sub(conversation.group("sender")))

    for pattern in (_THREADS_BY_USER_RE, _THREADS_INDEX_EVENT_RE, _THREADS_DELETE_RE):
        threads = pattern.match(path)
        if threads:
            return Decision("bound", threads.group("sub"))

    # /model/*, /domain, /status, /, health checks, anything Rasa adds later.
    return DENY
