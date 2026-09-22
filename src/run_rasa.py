# mypy: ignore_missing_imports = True
# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false

import asyncio
import contextvars
import inspect
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from inspect import isawaitable
from typing import Any, Optional, cast
from urllib.parse import urlsplit

import rasa  # type: ignore
import rasa.__main__ as rasa_main  # type: ignore
import rasa.core.run as core_run  # type: ignore
import requests  # type: ignore
from sanic import response  # type: ignore
from sanic_routing.exceptions import RouteExists  # type: ignore

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Matplotlib created a temporary config/cache directory*")

from src.request_identity_policy import WEBHOOK_PATH, has_required_role, required_identity  # noqa: E402
from src.thread_index import (  # noqa: E402
    apply_index_action,
    build_thread_list_from_payload,
    build_thread_list_response,
    next_thread_id_from_payload,
)
from src.thread_index_store import get_index_payload, set_index_payload  # noqa: E402

logger = logging.getLogger(__name__)


def _read_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None

    normalized = value.strip()
    return normalized or None


# Every request (except GET /version) must carry a real Keycloak access token
# as `Authorization: Bearer <token>`, verified via Keycloak's introspection
# endpoint, and its verified subject must match the user_sub/sender_id being
# acted on -- see src/request_identity_policy.py for the per-route rules and
# _enforce_request_identity below for the gate. There is no static shared
# secret: Rasa is started without --auth-token, so this gate is the only
# authentication in front of Rasa's built-in API. Reuses Webapp's own
# confidential client credentials (KEYCLOAK_CLIENT_ID/_SECRET) rather than a
# separate introspection client.
_KEYCLOAK_ISSUER = _read_env("KEYCLOAK_ISSUER")
_KEYCLOAK_CLIENT_ID = _read_env("KEYCLOAK_CLIENT_ID")
_KEYCLOAK_CLIENT_SECRET = _read_env("KEYCLOAK_CLIENT_SECRET")
if not (_KEYCLOAK_ISSUER and _KEYCLOAK_CLIENT_ID and _KEYCLOAK_CLIENT_SECRET):
    raise RuntimeError("KEYCLOAK_ISSUER, KEYCLOAK_CLIENT_ID and KEYCLOAK_CLIENT_SECRET are all required.")


def _introspect_token_sync(token: str) -> Optional[str]:
    """Verify a bearer token via Keycloak introspection; return the verified sub, or None."""
    if not (_KEYCLOAK_ISSUER and _KEYCLOAK_CLIENT_ID and _KEYCLOAK_CLIENT_SECRET):
        return None
    try:
        resp = requests.post(
            f"{_KEYCLOAK_ISSUER.rstrip('/')}/protocol/openid-connect/token/introspect",
            data={
                "token": token,
                "client_id": _KEYCLOAK_CLIENT_ID,
                "client_secret": _KEYCLOAK_CLIENT_SECRET,
            },
            timeout=5,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        logger.warning("Keycloak token introspection request failed", exc_info=True)
        return None

    if not payload.get("active") or not has_required_role(payload):
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) and sub else None


async def _verify_user_token(request) -> Optional[str]:
    """Extract and verify the Authorization: Bearer token; return the verified sub, or None."""
    token = _bearer_token(request)
    if not token:
        return None
    return await asyncio.to_thread(_introspect_token_sync, token)


# Bearer token of the request being served, verified by the gate below. Rasa's
# outgoing action-server calls send it on so Action can verify the same user;
# a contextvar because the action call happens deep inside Rasa's own message
# handling, far from any handler of ours, but within the same request task.
_verified_user_token: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "verified_user_token", default=None
)


def _bearer_token(request) -> str:
    auth_header = request.headers.get("Authorization", "") if hasattr(request, "headers") else ""
    return auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""


def _forward_user_token_to_action(action_endpoint: Any) -> None:
    """Make every request Rasa sends to the action server carry the current
    request's verified user token, replacing a static shared secret."""
    original_request = action_endpoint.request

    async def request_with_user_token(*args, **kwargs):
        token = _verified_user_token.get()
        if token:
            kwargs["headers"] = {**kwargs.get("headers", {}), "Authorization": f"Bearer {token}"}
        return await original_request(*args, **kwargs)

    action_endpoint.request = request_with_user_token


async def _enforce_request_identity(request):
    """Sanic on_request hook: the single authentication gate for every route.

    Returns a response to short-circuit the request, or None to let it
    through. A request is only let through if the policy allows it outright
    (GET /version, CORS preflight) or its verified Keycloak subject matches
    the user the path or webhook body says it acts for.
    """
    body_sender = None
    if request.method == "POST" and request.path == WEBHOOK_PATH:
        try:
            body = request.json if isinstance(request.json, dict) else {}
        except Exception:
            body = {}
        candidate = body.get("sender")
        body_sender = candidate if isinstance(candidate, str) and candidate else None

    decision = required_identity(request.method, request.path, body_sender)
    if decision.kind == "open":
        return None
    if decision.kind == "deny":
        return response.json({"error": "Forbidden"}, status=403)
    if decision.kind == "malformed":
        return response.json({"error": "Missing sender"}, status=400)

    verified_sub = await _verify_user_token(request)
    if not verified_sub:
        return response.json({"error": "Unauthorized"}, status=401)
    if verified_sub != decision.sub:
        return response.json({"error": "Forbidden: token subject does not match the requested user"}, status=403)
    _verified_user_token.set(_bearer_token(request))
    return None


def _with_build_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Overlay the image's build metadata (set by the Dockerfile from CI build
    args) on Rasa's own /version payload. Unset values are skipped, so a local
    run keeps whatever Rasa reported itself."""
    metadata = {
        "service": "rasa",
        "version": _read_env("RASA_VERSION"),
        "frameworkVersion": rasa.__version__,
        "commitSha": _read_env("RASA_COMMIT_SHA"),
        "imageTag": _read_env("RASA_IMAGE_TAG"),
        "buildDate": _read_env("RASA_BUILD_DATE"),
        "ssotVersion": _read_env("RASA_SSOT_VERSION"),
    }
    return {**payload, **{key: value for key, value in metadata.items() if value is not None}}


async def _add_build_metadata_to_version(request, resp) -> None:
    """Sanic on_response hook. Rasa core registers GET /version itself and
    answers with only its own version fields, so a handler of ours on that
    path can never be added (the route already exists); merge the build
    metadata into core's response instead."""
    if request.method != "GET" or request.path != "/version" or resp is None or resp.status != 200:
        return None
    try:
        payload = json.loads(resp.body)
    except (TypeError, ValueError):
        return None
    if isinstance(payload, dict):
        resp.body = json.dumps(_with_build_metadata(payload)).encode()
    return None


async def _hard_delete_tracker(tracker_store: Any, sender_id: str) -> bool:
    """Best-effort physical deletion of a tracker.

    First choice is always the store's own generic `.delete()` method, if it
    implements one -- that's the interface Rasa's own docs define for
    custom tracker stores (see
    https://rasa.com/docs/reference/integrations/tracker-stores/, "Custom
    Tracker Store"), so any future store built to that contract just works
    here with no changes needed.

    None of Rasa's *built-in* stores implement `.delete()` though (confirmed
    against both the installed package and current upstream main), so below
    that is one duck-typed fallback per built-in backend -- detected by
    attribute shape, not `isinstance`, so this doesn't need to import any of
    Rasa's internal store classes and keeps working even if their module
    paths move in a future Rasa version:
    - Redis (`.red`/`.redis` + `.key_prefix`): DEL the key directly.
    - SQL (`.session_scope` + `.SQLEvent`, e.g. Postgres/SQLite/Oracle):
      DELETE FROM events WHERE sender_id = ... via the store's own session.
    - Mongo (`.conversations`, a pymongo Collection): delete_many by
      sender_id field.
    - DynamoDB (`.db` with `.delete_item` + `.table_name`, distinguishing it
      from Mongo's own unrelated `.db` attribute): delete_item by the
      sender_id hash key.
    - InMemory (`.store`, a plain dict): pop the key. Only matters within
      this one process/worker and this Rasa run (never persisted or shared
      to begin with), but still worth clearing for consistency.
    Checked in roughly most-to-least-likely-in-a-real-deployment order.
    Returns whether deletion is known to have succeeded.

    `agent.tracker_store` is never the raw configured store -- Rasa always
    wraps it (at minimum in `AwaitableTrackerStore`, often also
    `FailSafeTrackerStore`), and both wrappers hold the real store under the
    same private `_tracker_store` attribute with no passthrough for any of
    the backend-specific attributes above. Unwrap down to the real store
    first, or every lookup below silently finds nothing on the wrapper and
    this always reports failure."""
    while hasattr(tracker_store, "_tracker_store"):
        tracker_store = tracker_store._tracker_store

    delete_fn = getattr(tracker_store, "delete", None)
    if callable(delete_fn):
        try:
            result = delete_fn(sender_id)
            if isawaitable(result):
                result = await cast(Any, result)
            if bool(result):
                return True
        except Exception:
            # Caught, not raised, so a store implementing .delete() badly
            # doesn't block falling through to the duck-typed backend
            # checks below -- but still logged, since a matched-but-broken
            # custom store is a real operator-relevant error, not just "not
            # this backend type" (see the module docstring's distinction).
            logger.warning("_hard_delete_tracker: custom .delete() raised for sender_id=%s", sender_id, exc_info=True)

    redis_client = getattr(tracker_store, "red", None) or getattr(tracker_store, "redis", None)
    if redis_client is not None:
        try:
            key_prefix = getattr(tracker_store, "key_prefix", "") or ""
            deleted = redis_client.delete(f"{key_prefix}{sender_id}")
            return bool(deleted)
        except Exception:
            logger.warning("_hard_delete_tracker: Redis DEL raised for sender_id=%s", sender_id, exc_info=True)

    session_scope = getattr(tracker_store, "session_scope", None)
    sql_event = getattr(tracker_store, "SQLEvent", None)
    if callable(session_scope) and sql_event is not None:
        try:
            with session_scope() as session:
                deleted_count = session.query(sql_event).filter(sql_event.sender_id == sender_id).delete()
                session.commit()
            return bool(deleted_count)
        except Exception:
            logger.warning("_hard_delete_tracker: SQL delete raised for sender_id=%s", sender_id, exc_info=True)

    conversations = getattr(tracker_store, "conversations", None)
    if conversations is not None and callable(getattr(conversations, "delete_many", None)):
        try:
            result = conversations.delete_many({"sender_id": sender_id})
            return bool(getattr(result, "deleted_count", 0))
        except Exception:
            logger.warning("_hard_delete_tracker: Mongo delete_many raised for sender_id=%s", sender_id, exc_info=True)

    dynamo_table = getattr(tracker_store, "db", None)
    if (
        dynamo_table is not None
        and hasattr(tracker_store, "table_name")
        and callable(getattr(dynamo_table, "delete_item", None))
    ):
        try:
            dynamo_table.delete_item(Key={"sender_id": sender_id})
            return True
        except Exception:
            logger.warning("_hard_delete_tracker: DynamoDB delete_item raised for sender_id=%s", sender_id, exc_info=True)

    memory_store = getattr(tracker_store, "store", None)
    if isinstance(memory_store, dict):
        if sender_id in memory_store:
            try:
                del memory_store[sender_id]
                return True
            except Exception:
                logger.warning("_hard_delete_tracker: in-memory delete raised for sender_id=%s", sender_id, exc_info=True)
        return False

    return False


def _install_custom_routes() -> None:
    original_configure_app = core_run.configure_app

    def configure_app_with_custom_routes(*args, **kwargs):
        app = original_configure_app(*args, **kwargs)
        endpoints = inspect.signature(original_configure_app).bind(*args, **kwargs).arguments.get("endpoints")
        if getattr(endpoints, "action", None) is not None:
            _forward_user_token_to_action(endpoints.action)

        def _safe_add(handler, path: str, methods: list[str]) -> None:
            try:
                app.add_route(handler, path, methods=methods)
            except RouteExists:
                pass

        async def _get_tracker_store() -> tuple[Any | None, Any | None]:
            agent = getattr(getattr(app, "ctx", None), "agent", None)
            if not agent:
                return None, response.json({"error": "Agent not initialized"}, status=500)
            tracker_store = getattr(agent, "tracker_store", None)
            if not tracker_store:
                return None, response.json({"error": "Tracker store not available"}, status=500)
            return cast(Any, tracker_store), None

        async def get_threads(request, user_sub: str):
            payload = get_index_payload(user_sub)
            threads = build_thread_list_from_payload(payload)
            return response.json(build_thread_list_response(threads), status=200)

        async def get_next_thread_id(request, user_sub: str):
            payload = get_index_payload(user_sub)
            return response.json(
                {
                    "next_thread_id": next_thread_id_from_payload(payload),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status=200,
            )

        async def post_index_event(request, user_sub: str):
            payload = request.json if isinstance(request.json, dict) else None
            if payload is None:
                return response.json({"error": "Invalid JSON"}, status=400)

            thread_id_raw = payload.get("thread_id")
            action = payload.get("action")
            name = payload.get("name", "")
            if thread_id_raw is None:
                return response.json({"error": "Missing or invalid thread_id, action"}, status=400)
            try:
                thread_id = int(str(thread_id_raw))
            except (TypeError, ValueError):
                thread_id = None

            if thread_id is None or action not in {"create", "rename", "delete"}:
                return response.json({"error": "Missing or invalid thread_id, action"}, status=400)

            current_payload = get_index_payload(user_sub)
            next_payload = apply_index_action(current_payload, thread_id, action, str(name))
            set_index_payload(user_sub, next_payload)

            threads = build_thread_list_from_payload(next_payload)
            thread_record = threads.get(thread_id)
            return response.json(
                {
                    "ok": True,
                    "action": action,
                    "thread": thread_record,
                    "next_thread_id": next_thread_id_from_payload(next_payload),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status=200,
            )

        async def delete_thread(request, user_sub: str, thread_id: str):
            """DELETE /threads/<user_sub>/thread/<thread_id> - Delete a thread and its tracker."""
            try:
                thread_id_int = int(thread_id)
            except (TypeError, ValueError):
                return response.json({"error": "Invalid thread_id"}, status=400)

            # Check the thread exists in the index first.
            current_payload = get_index_payload(user_sub)
            threads = build_thread_list_from_payload(current_payload)
            if thread_id_int not in threads:
                return response.json({"error": "Thread not found"}, status=404)

            tracker_store, err = await _get_tracker_store()
            if err:
                return err
            if tracker_store is None:
                return response.json({"error": "Tracker store not available"}, status=500)

            conversation_sender_id = f"{user_sub}:thread:{thread_id_int}"
            physically_deleted = await _hard_delete_tracker(tracker_store, conversation_sender_id)

            # Soft-mark as deleted in the index regardless of hard-delete outcome.
            next_payload = apply_index_action(current_payload, thread_id_int, "delete")
            set_index_payload(user_sub, next_payload)

            return response.json(
                {
                    "ok": True,
                    "thread_id": thread_id_int,
                    "physically_deleted": physically_deleted,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status=200,
            )

        async def delete_conversation_tracker(request, conversation_id: str):
            """DELETE /conversations/<conversation_id>/tracker - hard-delete a
            tracker directly by sender_id, independent of the thread index
            (unlike delete_thread above). For callers that intentionally
            don't register into the same user-facing thread index Webapp's
            real chat uses (e.g. CVaLab's own direct-to-Rasa debug chat,
            which deliberately avoids mixing its threads into a real user's
            actual Webapp thread list) but still need real deletion, not
            just an orphaned tracker. Same naming convention as the existing
            GET/PUT /conversations/<conversation_id>/tracker routes."""
            tracker_store, err = await _get_tracker_store()
            if err:
                return err
            if tracker_store is None:
                return response.json({"error": "Tracker store not available"}, status=500)

            physically_deleted = await _hard_delete_tracker(tracker_store, conversation_id)

            return response.json(
                {
                    "ok": True,
                    "sender_id": conversation_id,
                    "physically_deleted": physically_deleted,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status=200,
            )

        # The handlers above do no authentication of their own; this hook
        # runs before every route, built-in or custom.
        app.on_request(_enforce_request_identity)
        app.on_response(_add_build_metadata_to_version)

        _safe_add(get_threads, "/threads/by-user/<user_sub:str>", ["GET"])
        _safe_add(get_next_thread_id, "/threads/by-user/<user_sub:str>/next-id", ["GET"])
        _safe_add(post_index_event, "/threads/<user_sub:str>/index-event", ["POST"])
        _safe_add(delete_thread, "/threads/<user_sub:str>/thread/<thread_id:str>", ["DELETE"])
        _safe_add(delete_conversation_tracker, "/conversations/<conversation_id:path>/tracker", ["DELETE"])

        return app

    core_run.configure_app = configure_app_with_custom_routes


def _resolve_endpoints_file() -> str:
    explicit_file = _read_env("RASA_ENDPOINTS_FILE")
    if explicit_file:
        return explicit_file

    backend = (_read_env("RASA_TRACKER_STORE_BACKEND") or "redis").lower()
    presets = {
        "memory": "src/core/endpoints.memory.yml",
        "redis": "src/core/endpoints.redis.yml",
        "sql": "src/core/endpoints.sql.yml",
        "mongo": "src/core/endpoints.mongo.yml",
        "dynamo": "src/core/endpoints.dynamo.yml",
        "sqlite": "src/core/endpoints.sqlite.yml",
    }
    return presets.get(backend, "src/core/endpoints.redis.yml")


def _resolve_cors() -> Optional[str]:
    cors = _read_env("RASA_CORS")
    if cors is None:
        return None

    if "*" in cors:
        raise RuntimeError("RASA_CORS must use an explicit origin; wildcard values are not allowed.")

    if "," in cors or ";" in cors:
        raise RuntimeError("RASA_CORS must be a single explicit origin.")

    parsed = urlsplit(cors)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("RASA_CORS must be an http(s) origin such as https://example.com.")

    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise RuntimeError("RASA_CORS must be a bare origin without path, query, or fragment.")

    return f"{parsed.scheme}://{parsed.netloc}"


def main() -> None:
    _install_custom_routes()
    endpoints_file = _resolve_endpoints_file()
    cors = _resolve_cors()
    # Docker runs this entrypoint without CLI args by default; in that case,
    # provide sensible defaults and resolve the backend endpoints from env.
    if len(sys.argv) == 1:
        args = [
            "run",
            "--enable-api",
            "--model",
            "models",
            "--endpoints",
            endpoints_file,
            "--request-timeout",
            os.getenv("RASA_REQUEST_TIMEOUT", "300"),
            "--response-timeout",
            os.getenv("RASA_RESPONSE_TIMEOUT", "300"),
        ]
        if cors:
            args.extend(["--cors", cors])
        sys.argv.extend(args)
    rasa_main.main()


if __name__ == "__main__":
    main()
