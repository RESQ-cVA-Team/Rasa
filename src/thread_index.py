"""Thread index helpers for Rasa custom thread routes."""

import json
from datetime import datetime
from typing import Any, Dict


def get_thread_index_tracker_id(user_sub: str) -> str:
    """Return the dedicated tracker ID used to store a user's thread index."""
    return f"{user_sub}:threads:index"


def build_thread_list_from_payload(payload: Any) -> Dict[int, Dict[str, Any]]:
    """Build active thread records from a JSON payload persisted in the index tracker."""
    if not payload:
        return {}

    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except (json.JSONDecodeError, TypeError):
        return {}

    if not isinstance(data, dict):
        return {}

    threads: Dict[int, Dict[str, Any]] = {}
    for raw_id, raw_thread in data.items():
        try:
            thread_id = int(raw_id)
        except (TypeError, ValueError):
            continue

        if not isinstance(raw_thread, dict):
            continue

        action = raw_thread.get("action", "create")
        if action == "delete":
            continue

        threads[thread_id] = {
            "id": thread_id,
            "name": raw_thread.get("name", ""),
            "created_at": raw_thread.get("created_at", datetime.utcnow().isoformat()),
            "updated_at": raw_thread.get("timestamp", datetime.utcnow().isoformat()),
            "deleted": False,
        }

    return threads


def build_thread_list_response(threads: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Serialize thread records into API response format."""
    ordered_threads = sorted(
        threads.values(),
        key=lambda thread: thread.get("updated_at", ""),
        reverse=True,
    )

    return {
        "threads": ordered_threads,
        "count": len(ordered_threads),
        "timestamp": datetime.utcnow().isoformat(),
    }
