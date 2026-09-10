# mypy: ignore_missing_imports = True
# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false

import asyncio
import logging
import os
import re
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


def _env_flag(name: str, default: bool) -> bool:
    value = _read_env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


# Phase 2 of the cross-service auth redesign: RASA_AUTH_TOKEN (checked by
# _authorized below, and by Rasa core's own global --auth-token middleware)
# proves only that *a* trusted service is calling, never *which* user a
# request is for -- user_sub/sender_id are otherwise just caller-supplied
# strings, trusted as-is. This verifies the real Keycloak access token
# Webapp forwards as `Authorization: Bearer <token>` (see
# Webapp/src/lib/rasaConfig.ts's withUserBearerHeader) via Keycloak's
# introspection endpoint, and requires the verified subject to match the
# user_sub/sender_id being acted on. Reuses Webapp's own confidential
# client credentials (KEYCLOAK_CLIENT_ID/_SECRET) rather than requiring a
# separate dedicated introspection client -- Keycloak's introspection
# endpoint only needs a valid confidential client, not a purpose-specific
# one. Gated behind REQUIRE_USER_TOKEN_VERIFICATION (default off) for a
# rollout window during which Webapp may not yet forward the header on
# every deployed instance.
_KEYCLOAK_ISSUER = _read_env("KEYCLOAK_ISSUER")
_KEYCLOAK_CLIENT_ID = _read_env("KEYCLOAK_CLIENT_ID")
_KEYCLOAK_CLIENT_SECRET = _read_env("KEYCLOAK_CLIENT_SECRET")
_REQUIRE_USER_TOKEN_VERIFICATION = _env_flag("REQUIRE_USER_TOKEN_VERIFICATION", default=False)
if _REQUIRE_USER_TOKEN_VERIFICATION and not (_KEYCLOAK_ISSUER and _KEYCLOAK_CLIENT_ID and _KEYCLOAK_CLIENT_SECRET):
    raise RuntimeError(
        "KEYCLOAK_ISSUER, KEYCLOAK_CLIENT_ID and KEYCLOAK_CLIENT_SECRET are all required when "
        "REQUIRE_USER_TOKEN_VERIFICATION is enabled."
    )

_SENDER_THREAD_SUFFIX_RE = re.compile(r"^(.*):thread:(\d+)$")


def _sender_sub(sender_id: str) -> str:
    """Strip the `:thread:<id>` suffix, mirroring rasaSender.ts's parseRasaSenderId."""
    match = _SENDER_THREAD_SUFFIX_RE.match(sender_id)
    return match.group(1) if match else sender_id


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

    if not payload.get("active"):
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) and sub else None


async def _verify_user_token(request) -> Optional[str]:
    """Extract and verify the Authorization: Bearer token; return the verified sub, or None."""
    auth_header = request.headers.get("Authorization", "") if hasattr(request, "headers") else ""
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:].strip()
    if not token:
        return None
    return await asyncio.to_thread(_introspect_token_sync, token)


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

        async def version(_):
            return response.json(
                {
                    "service": "rasa",
                    "version": _read_env("RASA_VERSION"),
                    "frameworkVersion": rasa.__version__,
                    "commitSha": _read_env("RASA_COMMIT_SHA"),
                    "imageTag": _read_env("RASA_IMAGE_TAG"),
                    "buildDate": _read_env("RASA_BUILD_DATE"),
                    "ssotVersion": _read_env("RASA_SSOT_VERSION"),
                },
                status=200,
            )

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

        def _authorized(request) -> bool:
            expected = _read_env("RASA_AUTH_TOKEN")
            if not expected:
                return True

            query_token = request.args.get("token") if hasattr(request, "args") else None
            auth_header = request.headers.get("Authorization", "") if hasattr(request, "headers") else ""
            header_token = auth_header[7:] if auth_header.startswith("Bearer ") else None
            return (query_token or header_token) == expected

        async def _check_user_identity(request, claimed_sub: str):
            """When REQUIRE_USER_TOKEN_VERIFICATION is on, verify the caller's
            Bearer token and require it to match claimed_sub. Returns an error
            response to return immediately, or None if the caller may proceed.
            """
            if not _REQUIRE_USER_TOKEN_VERIFICATION:
                return None
            verified_sub = await _verify_user_token(request)
            if not verified_sub:
                return response.json({"error": "Unauthorized"}, status=401)
            if verified_sub != claimed_sub:
                return response.json(
                    {"error": "Forbidden: token subject does not match the requested user"}, status=403
                )
            return None

        async def get_threads(request, user_sub: str):
            if not _authorized(request):
                return response.json({"error": "Unauthorized"}, status=401)
            identity_err = await _check_user_identity(request, user_sub)
            if identity_err:
                return identity_err

            payload = get_index_payload(user_sub)
            threads = build_thread_list_from_payload(payload)
            return response.json(build_thread_list_response(threads), status=200)

        async def get_next_thread_id(request, user_sub: str):
            if not _authorized(request):
                return response.json({"error": "Unauthorized"}, status=401)
            identity_err = await _check_user_identity(request, user_sub)
            if identity_err:
                return identity_err

            payload = get_index_payload(user_sub)
            return response.json(
                {
                    "next_thread_id": next_thread_id_from_payload(payload),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status=200,
            )

        async def post_index_event(request, user_sub: str):
            if not _authorized(request):
                return response.json({"error": "Unauthorized"}, status=401)
            identity_err = await _check_user_identity(request, user_sub)
            if identity_err:
                return identity_err

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
            if not _authorized(request):
                return response.json({"error": "Unauthorized"}, status=401)
            identity_err = await _check_user_identity(request, user_sub)
            if identity_err:
                return identity_err

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
            if not _authorized(request):
                return response.json({"error": "Unauthorized"}, status=401)
            identity_err = await _check_user_identity(request, _sender_sub(conversation_id))
            if identity_err:
                return identity_err

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

        @app.on_request
        async def _verify_webhook_identity(request):
            # The standard REST webhook is Rasa core's own built-in channel
            # route, not one of the custom routes above -- this is the only
            # hook point available to apply the same jobId-era identity check
            # to it. `sender` in the POST body is otherwise exactly as
            # caller-supplied/unverified as user_sub is on the custom routes.
            if request.path != "/webhooks/rest/webhook" or not _REQUIRE_USER_TOKEN_VERIFICATION:
                return None
            try:
                body = request.json if isinstance(request.json, dict) else {}
            except Exception:
                body = {}
            claimed_sender = body.get("sender")
            if not isinstance(claimed_sender, str) or not claimed_sender:
                return None  # malformed body -- let the route's own validation reject it
            return await _check_user_identity(request, _sender_sub(claimed_sender))

        _safe_add(version, "/version", ["GET"])
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


def _resolve_auth_token() -> str:
    require_auth = _env_flag("RASA_REQUIRE_AUTH_TOKEN", default=True)
    token = _read_env("RASA_AUTH_TOKEN")
    if require_auth and not token:
        raise RuntimeError("RASA_AUTH_TOKEN is required when RASA_REQUIRE_AUTH_TOKEN is enabled. Set RASA_AUTH_TOKEN or set RASA_REQUIRE_AUTH_TOKEN=false only for local debugging.")
    return token or ""


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
    auth_token = _resolve_auth_token()
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
        if auth_token:
            args.extend(["--auth-token", auth_token])
        if cors:
            args.extend(["--cors", cors])
        sys.argv.extend(args)
    rasa_main.main()


if __name__ == "__main__":
    main()
