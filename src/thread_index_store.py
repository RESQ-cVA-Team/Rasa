"""Redis-backed storage for the per-user thread index.

This used to piggyback on Rasa's conversation-event tracker store: every
thread action (create/rename/delete) re-appended the *entire* current index
payload as a new tracker event, and the tracker's full event history is what
gets read/serialized on every subsequent access. That meant storage grew
quadratically with the number of actions taken -- N actions produced N
snapshots, each O(current thread count), for O(N^2) total bytes -- and only
the single latest snapshot was ever actually used. A single Rasa test user
subjected to a few hundred CVaLab-driven thread creations grew this to over
80MB, which then made every thread-list/thread-create operation on that
account pay the cost of deserializing the entire accumulated history.

This module stores exactly one JSON value per user, overwritten in place.
Size is always O(current thread count), independent of how many actions were
ever taken.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import redis  # type: ignore[import-untyped]

_KEY_PREFIX = "cva:thread-index:"

_client: Optional["redis.Redis"] = None


def _get_client() -> "redis.Redis":
    global _client
    if _client is None:
        host = os.environ.get("TRACKER_STORE_URL", "redis")
        port = int(os.environ.get("TRACKER_STORE_PORT", "6379") or "6379")
        db = int(os.environ.get("TRACKER_STORE_DB", "0") or "0")
        password = os.environ.get("REDIS_PASSWORD") or None
        _client = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)
    return _client


def _key_for(user_sub: str) -> str:
    return f"{_KEY_PREFIX}{user_sub}"


def get_index_payload(user_sub: str) -> Dict[str, Any]:
    raw = _get_client().get(_key_for(user_sub))
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def set_index_payload(user_sub: str, payload: Dict[str, Any]) -> None:
    _get_client().set(_key_for(user_sub), json.dumps(payload))
