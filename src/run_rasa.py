import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlsplit


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Matplotlib created a temporary config/cache directory*")

import rasa
import rasa.__main__ as rasa_main
import rasa.core.run as core_run
from sanic import response
from sanic_routing.exceptions import RouteExists

from src.thread_index import (
    apply_index_action,
    build_thread_list_from_payload,
    build_thread_list_response,
    extract_index_payload_from_events,
    get_thread_index_tracker_id,
    next_thread_id_from_payload,
    serialize_index_payload,
)


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

        async def _get_tracker_store():
            agent = getattr(getattr(app, "ctx", None), "agent", None)
            if not agent:
                return None, response.json({"error": "Agent not initialized"}, status=500)
            tracker_store = getattr(agent, "tracker_store", None)
            if not tracker_store:
                return None, response.json({"error": "Tracker store not available"}, status=500)
            return tracker_store, None

        def _authorized(request) -> bool:
            expected = _read_env("RASA_AUTH_TOKEN")
            if not expected:
                return True

            query_token = request.args.get("token") if hasattr(request, "args") else None
            auth_header = request.headers.get("Authorization", "") if hasattr(request, "headers") else ""
            header_token = auth_header[7:] if auth_header.startswith("Bearer ") else None
            return (query_token or header_token) == expected

        async def get_threads(request, user_sub: str):
            if not _authorized(request):
                return response.json({"error": "Unauthorized"}, status=401)

            tracker_store, err = await _get_tracker_store()
            if err:
                return err

            tracker = await tracker_store.retrieve(get_thread_index_tracker_id(user_sub))
            if not tracker:
                return response.json(build_thread_list_response({}), status=200)

            payload = extract_index_payload_from_events(list(tracker.events))
            threads = build_thread_list_from_payload(payload)
            return response.json(build_thread_list_response(threads), status=200)

        async def get_next_thread_id(request, user_sub: str):
            if not _authorized(request):
                return response.json({"error": "Unauthorized"}, status=401)

            tracker_store, err = await _get_tracker_store()
            if err:
                return err

            tracker = await tracker_store.retrieve(get_thread_index_tracker_id(user_sub))
            payload = extract_index_payload_from_events(list(tracker.events)) if tracker else {}
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

            payload = request.json if isinstance(request.json, dict) else None
            if payload is None:
                return response.json({"error": "Invalid JSON"}, status=400)

            thread_id_raw = payload.get("thread_id")
            action = payload.get("action")
            name = payload.get("name", "")
            try:
                thread_id = int(thread_id_raw)
            except (TypeError, ValueError):
                thread_id = None

            if thread_id is None or action not in {"create", "rename", "delete"}:
                return response.json({"error": "Missing or invalid thread_id, action"}, status=400)

            tracker_store, err = await _get_tracker_store()
            if err:
                return err

            sender_id = get_thread_index_tracker_id(user_sub)
            tracker = await tracker_store.retrieve(sender_id)
            if tracker is None:
                from rasa.shared.core.trackers import DialogueStateTracker

                tracker = DialogueStateTracker(sender_id, [])

            current_payload = extract_index_payload_from_events(list(tracker.events))
            next_payload = apply_index_action(current_payload, thread_id, action, str(name))

            from rasa.shared.core.events import UserUttered

            tracker.update(
                UserUttered(
                    text=serialize_index_payload(next_payload),
                    intent={"name": "__thread_index_update__", "confidence": 1.0},
                    entities=[],
                )
            )
            await tracker_store.save(tracker)

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

            try:
                thread_id_int = int(thread_id)
            except (TypeError, ValueError):
                return response.json({"error": "Invalid thread_id"}, status=400)

            tracker_store, err = await _get_tracker_store()
            if err:
                return err

            # Check the thread exists in the index first.
            index_sender_id = get_thread_index_tracker_id(user_sub)
            index_tracker = await tracker_store.retrieve(index_sender_id)
            if index_tracker is None:
                return response.json({"error": "Thread not found"}, status=404)

            current_payload = extract_index_payload_from_events(list(index_tracker.events))
            threads = build_thread_list_from_payload(current_payload)
            if thread_id_int not in threads:
                return response.json({"error": "Thread not found"}, status=404)

            # Attempt hard-delete of the conversation tracker for this thread.
            conversation_sender_id = f"{user_sub}:thread:{thread_id_int}"
            physically_deleted = False
            delete_fn = getattr(tracker_store, "delete", None)
            if callable(delete_fn):
                try:
                    result = delete_fn(conversation_sender_id)
                    if hasattr(result, "__await__"):
                        result = await result
                    physically_deleted = bool(result)
                except Exception:
                    pass

            if not physically_deleted:
                # Fallback: if the store is Redis-backed, attempt direct key deletion.
                redis_client = getattr(tracker_store, "red", None) or getattr(tracker_store, "redis", None)
                if redis_client is not None:
                    try:
                        key_prefix = getattr(tracker_store, "key_prefix", "") or ""
                        deleted = redis_client.delete(f"{key_prefix}{conversation_sender_id}")
                        physically_deleted = bool(deleted)
                    except Exception:
                        pass

            # Soft-mark as deleted in the index tracker regardless of hard-delete outcome.
            next_payload = apply_index_action(current_payload, thread_id_int, "delete")

            from rasa.shared.core.events import UserUttered

            index_tracker.update(
                UserUttered(
                    text=serialize_index_payload(next_payload),
                    intent={"name": "__thread_index_update__", "confidence": 1.0},
                    entities=[],
                )
            )
            await tracker_store.save(index_tracker)

            return response.json(
                {
                    "ok": True,
                    "thread_id": thread_id_int,
                    "physically_deleted": physically_deleted,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status=200,
            )

        _safe_add(version, "/version", ["GET"])
        _safe_add(get_threads, "/threads/by-user/<user_sub:str>", ["GET"])
        _safe_add(get_next_thread_id, "/threads/by-user/<user_sub:str>/next-id", ["GET"])
        _safe_add(post_index_event, "/threads/<user_sub:str>/index-event", ["POST"])
        _safe_add(delete_thread, "/threads/<user_sub:str>/thread/<thread_id:str>", ["DELETE"])

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
        raise RuntimeError(
            "RASA_AUTH_TOKEN is required when RASA_REQUIRE_AUTH_TOKEN is enabled. "
            "Set RASA_AUTH_TOKEN or set RASA_REQUIRE_AUTH_TOKEN=false only for local debugging."
        )
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