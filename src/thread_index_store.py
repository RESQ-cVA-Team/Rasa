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

Every change is a read-modify-write of that value inside a WATCH/MULTI
transaction, so concurrent changes for one user are retried instead of
overwriting each other or handing out the same thread id twice.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional, Tuple

import redis  # type: ignore[import-untyped]

from src.thread_index import apply_index_action, build_thread_list_from_payload, next_thread_id_from_payload

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


def _decode(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def get_index_payload(user_sub: str) -> Dict[str, Any]:
    return _decode(_get_client().get(_key_for(user_sub)))


def _update(user_sub: str, change: Callable[[Dict[str, Any]], Tuple[Optional[Dict[str, Any]], Any]]) -> Any:
    """Runs `change` on the current index and writes the payload it returns
    (None writes nothing), retrying if another write lands in between."""
    key = _key_for(user_sub)

    def run(pipe: Any) -> Any:
        next_payload, result = change(_decode(pipe.get(key)))
        pipe.multi()
        if next_payload is not None:
            pipe.set(key, json.dumps(next_payload))
        return result

    return _get_client().transaction(run, key, value_from_callable=True)


def create_index_thread(user_sub: str, name: str) -> Dict[str, Any]:
    """Allocates the next thread id and records the thread in one step."""

    def change(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        thread_id = next_thread_id_from_payload(payload)
        next_payload = apply_index_action(payload, thread_id, "create", name.strip() or f"Conversation {thread_id}")
        return next_payload, build_thread_list_from_payload(next_payload)[thread_id]

    return _update(user_sub, change)


def rename_index_thread(user_sub: str, thread_id: int, name: str) -> Optional[Dict[str, Any]]:
    """None when the thread does not exist or was deleted."""

    def change(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if thread_id not in build_thread_list_from_payload(payload):
            return None, None
        next_payload = apply_index_action(payload, thread_id, "rename", name)
        return next_payload, build_thread_list_from_payload(next_payload)[thread_id]

    return _update(user_sub, change)


def delete_index_thread(user_sub: str, thread_id: int) -> bool:
    """Marks the thread deleted; False when it was not there."""

    def change(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], bool]:
        if thread_id not in build_thread_list_from_payload(payload):
            return None, False
        return apply_index_action(payload, thread_id, "delete"), True

    return _update(user_sub, change)
