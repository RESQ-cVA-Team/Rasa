import os
import sys
import warnings
from datetime import datetime
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

from .thread_index import (
    build_thread_list_from_payload,
    build_thread_list_response,
    get_thread_index_tracker_id,
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
    """Install custom /version and thread index routes."""
    original_configure_app = core_run.configure_app

    def configure_app_with_custom_routes(*args, **kwargs):
        app = original_configure_app(*args, **kwargs)

        # === /version route ===
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

        try:
            app.add_route(version, "/version", methods=["GET"])
        except RouteExists:
            pass

        # === Thread index routes ===
        def _require_auth(request):
            """Check auth token from query parameter or header."""
            token = _read_env("RASA_AUTH_TOKEN")
            if not token:
                return True  # Auth disabled, allow all
            
            query_token = request.args.get("token") if hasattr(request, "args") else None
            header_token = None
            if hasattr(request, "headers"):
                auth_header = request.headers.get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    header_token = auth_header[7:]
            
            provided_token = query_token or header_token
            return provided_token == token

        def _load_index_data(index_tracker) -> dict:
            import json

            marker = "__thread_index__"
            for event in reversed(list(index_tracker.events)):
                text = getattr(event, "text", None)
                if text and text.startswith(marker):
                    try:
                        return json.loads(text[len(marker):])
                    except Exception:
                        return {}
            return {}

        def _next_thread_id_from_index_data(index_data: dict) -> int:
            max_thread_id = 0
            for raw_id in index_data.keys():
                try:
                    parsed = int(raw_id)
                except Exception:
                    continue
                if parsed > max_thread_id:
                    max_thread_id = parsed
            return max_thread_id + 1

        async def _apply_index_event(
            tracker_store,
            user_sub: str,
            thread_id: int,
            action: str,
            name: str,
        ) -> None:
            import json
            from rasa.shared.core.events import UserUttered
            from rasa.shared.core.trackers import DialogueStateTracker

            index_sender_id = get_thread_index_tracker_id(user_sub)
            index_tracker = await tracker_store.retrieve(index_sender_id)
            if not index_tracker:
                index_tracker = DialogueStateTracker(index_sender_id, [])

            current_data = _load_index_data(index_tracker)
            thread_key = str(thread_id)

            if action == "create":
                current_data[thread_key] = {
                    "id": thread_id,
                    "name": name,
                    "action": "create",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            elif action == "rename":
                if thread_key not in current_data:
                    current_data[thread_key] = {"id": thread_id}
                current_data[thread_key].update(
                    {
                        "name": name,
                        "action": "rename",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            elif action == "delete":
                if thread_key not in current_data:
                    current_data[thread_key] = {"id": thread_id}
                current_data[thread_key].update(
                    {
                        "action": "delete",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            marker = "__thread_index__"
            index_event = UserUttered(
                text=f"{marker}{json.dumps(current_data)}",
                intent={"name": "__thread_index_update__", "confidence": 1.0},
                entities=[],
            )
            index_tracker.update(index_event)
            await tracker_store.save(index_tracker)

        async def _hard_delete_tracker_if_supported(tracker_store, sender_id: str) -> bool:
            """Attempt physical tracker deletion using backend capabilities when available."""
            try:
                delete_fn = getattr(tracker_store, "delete", None)
                if callable(delete_fn):
                    result = delete_fn(sender_id)
                    if hasattr(result, "__await__"):
                        await result
                    return True
            except Exception:
                pass

            # RedisTrackerStore exposes the raw redis client and key prefix.
            try:
                redis_client = getattr(tracker_store, "red", None)
                key_prefix = getattr(tracker_store, "key_prefix", None)
                if redis_client is not None and isinstance(key_prefix, str):
                    deleted = redis_client.delete(f"{key_prefix}{sender_id}")
                    return bool(deleted)
            except Exception:
                pass

            return False

        async def get_threads(request, user_sub: str):
            """GET /threads/by-user/<user_sub> - List threads for user."""
            try:
                if not _require_auth(request):
                    return response.json({"error": "Unauthorized"}, status=401)

                index_sender_id = get_thread_index_tracker_id(user_sub)

                if not hasattr(app.ctx, "agent") or not app.ctx.agent:
                    return response.json(
                        {"error": "Agent not initialized"},
                        status=500,
                    )

                tracker_store = app.ctx.agent.tracker_store
                if not tracker_store:
                    return response.json(
                        {"error": "Tracker store not available"},
                        status=500,
                    )

                index_tracker = await tracker_store.retrieve(index_sender_id)
                if not index_tracker:
                    return response.json(build_thread_list_response({}), status=200)

                payload = None
                marker = "__thread_index__"
                for event in reversed(list(index_tracker.events)):
                    text = getattr(event, "text", None)
                    if text and text.startswith(marker):
                        payload = text[len(marker):]
                        break

                threads = build_thread_list_from_payload(payload) if payload else {}
                result = build_thread_list_response(threads)

                return response.json(result, status=200)

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error retrieving thread index: {e}", exc_info=True)
                return response.json(
                    {"error": f"Internal server error: {str(e)}"},
                    status=500,
                )

        async def get_next_thread_id(request, user_sub: str):
            """GET /threads/by-user/<user_sub>/next-id - Next monotonic thread id."""
            try:
                if not _require_auth(request):
                    return response.json({"error": "Unauthorized"}, status=401)

                if not hasattr(app.ctx, "agent") or not app.ctx.agent:
                    return response.json(
                        {"error": "Agent not initialized"},
                        status=500,
                    )

                tracker_store = app.ctx.agent.tracker_store
                if not tracker_store:
                    return response.json(
                        {"error": "Tracker store not available"},
                        status=500,
                    )

                index_sender_id = get_thread_index_tracker_id(user_sub)
                index_tracker = await tracker_store.retrieve(index_sender_id)
                if not index_tracker:
                    return response.json(
                        {"next_thread_id": 1, "timestamp": datetime.utcnow().isoformat()},
                        status=200,
                    )

                index_data = _load_index_data(index_tracker)
                next_thread_id = _next_thread_id_from_index_data(index_data)

                return response.json(
                    {
                        "next_thread_id": next_thread_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    status=200,
                )

            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Error computing next thread id: {e}", exc_info=True)
                return response.json(
                    {"error": f"Internal server error: {str(e)}"},
                    status=500,
                )

        async def post_index_event(request, user_sub: str):
            """POST /threads/<user_sub>/index-event - Record index metadata event."""
            try:
                if not _require_auth(request):
                    return response.json({"error": "Unauthorized"}, status=401)

                try:
                    payload = request.json
                except Exception:
                    return response.json({"error": "Invalid JSON"}, status=400)

                thread_id = payload.get("thread_id")
                name = payload.get("name", "")
                action = payload.get("action")

                if thread_id is None or action not in {"create", "rename", "delete"}:
                    return response.json(
                        {"error": "Missing or invalid thread_id, action"},
                        status=400,
                    )

                if not hasattr(app.ctx, "agent") or not app.ctx.agent:
                    return response.json(
                        {"error": "Agent not initialized"},
                        status=500,
                    )

                tracker_store = app.ctx.agent.tracker_store
                if not tracker_store:
                    return response.json(
                        {"error": "Tracker store not available"},
                        status=500,
                    )

                await _apply_index_event(
                    tracker_store=tracker_store,
                    user_sub=user_sub,
                    thread_id=int(thread_id),
                    action=action,
                    name=name,
                )

                return response.json(
                    {
                        "success": True,
                        "thread_id": thread_id,
                        "action": action,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    status=201,
                )

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error recording index event: {e}", exc_info=True)
                return response.json(
                    {"error": f"Internal server error: {str(e)}"},
                    status=500,
                )

        async def delete_thread(request, user_sub: str, thread_id: str):
            """DELETE /threads/<user_sub>/thread/<thread_id> - Purge tracker and hide thread."""
            try:
                if not _require_auth(request):
                    return response.json({"error": "Unauthorized"}, status=401)

                try:
                    parsed_thread_id = int(thread_id)
                except Exception:
                    return response.json({"error": "Invalid thread_id"}, status=400)

                if parsed_thread_id <= 0:
                    return response.json({"error": "Invalid thread_id"}, status=400)

                if not hasattr(app.ctx, "agent") or not app.ctx.agent:
                    return response.json(
                        {"error": "Agent not initialized"},
                        status=500,
                    )

                tracker_store = app.ctx.agent.tracker_store
                if not tracker_store:
                    return response.json(
                        {"error": "Tracker store not available"},
                        status=500,
                    )

                from rasa.shared.core.trackers import DialogueStateTracker

                sender_id = f"{user_sub}:thread:{parsed_thread_id}"
                physically_deleted = await _hard_delete_tracker_if_supported(tracker_store, sender_id)

                if not physically_deleted:
                    # Fallback for stores without delete support.
                    empty_tracker = DialogueStateTracker(sender_id, [])
                    await tracker_store.save(empty_tracker)

                purged = False
                try:
                    stored_tracker = await tracker_store.retrieve_full_tracker(sender_id)
                    purged = stored_tracker is None or len(stored_tracker.events) == 0
                except Exception:
                    purged = False

                await _apply_index_event(
                    tracker_store=tracker_store,
                    user_sub=user_sub,
                    thread_id=parsed_thread_id,
                    action="delete",
                    name="",
                )

                return response.json(
                    {
                        "success": True,
                        "thread_id": parsed_thread_id,
                        "purged": purged,
                    },
                    status=200,
                )
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Error purging thread: {e}", exc_info=True)
                return response.json(
                    {"error": f"Internal server error: {str(e)}"},
                    status=500,
                )

        try:
            app.add_route(get_threads, "/threads/by-user/<user_sub>", methods=["GET"])
        except RouteExists:
            pass

        try:
            app.add_route(get_next_thread_id, "/threads/by-user/<user_sub>/next-id", methods=["GET"])
        except RouteExists:
            pass

        try:
            app.add_route(post_index_event, "/threads/<user_sub>/index-event", methods=["POST"])
        except RouteExists:
            pass

        try:
            app.add_route(delete_thread, "/threads/<user_sub>/thread/<thread_id>", methods=["DELETE"])
        except RouteExists:
            pass

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
