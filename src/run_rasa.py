import os
import sys
import warnings
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

from thread_index import (
    build_thread_list_from_events,
    build_thread_list_response,
    create_thread_metadata_event,
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


def _install_version_route() -> None:
    original_configure_app = core_run.configure_app

    def configure_app_with_version(*args, **kwargs):
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

        # Rasa can call configure_app multiple times during startup. Avoid
        # crashing when /version is already present.
        try:
            app.add_route(version, "/version", methods=["GET"])
        except RouteExists:
            pass

        return app

    core_run.configure_app = configure_app_with_version


def _install_thread_index_routes() -> None:
    """Install custom routes for thread index management."""
    original_configure_app = core_run.configure_app

    def configure_app_with_thread_routes(*args, **kwargs):
        app = original_configure_app(*args, **kwargs)

        def _require_auth(request):
            """Check auth token from query parameter or header."""
            token = _read_env("RASA_AUTH_TOKEN")
            if not token:
                return None  # Auth disabled
            
            query_token = request.args.get("token") if hasattr(request, "args") else None
            header_token = None
            if hasattr(request, "headers"):
                auth_header = request.headers.get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    header_token = auth_header[7:]
            
            provided_token = query_token or header_token
            return provided_token == token

        async def get_threads(request, user_sub: str):
            """GET /threads/by-user/<user_sub> - List threads for user."""
            try:
                # Check auth
                if not _require_auth(request):
                    return response.json({"error": "Unauthorized"}, status=401)

                # Get index tracker ID
                index_sender_id = get_thread_index_tracker_id(user_sub)

                # Access tracker store from app context
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

                # Retrieve index tracker
                index_tracker = await tracker_store.retrieve(index_sender_id)
                if not index_tracker:
                    # No index tracker yet (first time), return empty list
                    return response.json(build_thread_list_response({}), status=200)

                # Parse events and build thread list
                events = index_tracker.events if hasattr(index_tracker, "events") else []
                threads = build_thread_list_from_events(events)
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

        async def post_index_event(request, user_sub: str):
            """POST /threads/<user_sub>/index-event - Record index metadata event."""
            try:
                # Check auth
                if not _require_auth(request):
                    return response.json({"error": "Unauthorized"}, status=401)

                # Parse request body
                try:
                    payload = request.json
                except Exception:
                    return response.json({"error": "Invalid JSON"}, status=400)

                thread_id = payload.get("thread_id")
                name = payload.get("name", "")
                action = payload.get("action")  # "create", "rename", "delete"

                if thread_id is None or action not in {"create", "rename", "delete"}:
                    return response.json(
                        {"error": "Missing or invalid thread_id, action"},
                        status=400,
                    )

                # Get index tracker ID
                index_sender_id = get_thread_index_tracker_id(user_sub)

                # Access tracker store
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

                # Retrieve or create index tracker
                index_tracker = await tracker_store.retrieve(index_sender_id)
                if not index_tracker:
                    # Create new tracker
                    from rasa.core.trackers import DialogueStateTracker
                    index_tracker = DialogueStateTracker.new_session(index_sender_id)

                # Create and append metadata event
                event_dict = create_thread_metadata_event(thread_id, name, action)
                
                # Create custom event and append to tracker
                from rasa.core.events import Event
                # For now, we'll create a simple custom event object
                class ThreadMetadataEvent(Event):
                    type_name = "thread_metadata_update"
                    
                    def __init__(self, thread_id, name, action, timestamp=None):
                        super().__init__(timestamp)
                        self.thread_id = thread_id
                        self.name = name
                        self.action = action
                        self.parse_data = {
                            "thread_id": thread_id,
                            "name": name,
                            "action": action,
                            "timestamp": timestamp or self.timestamp.isoformat(),
                        }
                    
                    def as_dict(self):
                        d = super().as_dict()
                        d.update(self.parse_data)
                        return d

                event = ThreadMetadataEvent(thread_id, name, action)
                index_tracker.apply_latest_bot_action(index_tracker.slots)
                index_tracker.events.append(event)

                # Save updated tracker
                await tracker_store.save(index_tracker)

                return response.json(
                    {
                        "success": True,
                        "thread_id": thread_id,
                        "action": action,
                        "timestamp": event.timestamp.isoformat(),
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

        # Add routes, catching RouteExists for idempotency
        try:
            app.add_route(get_threads, "/threads/by-user/<user_sub>", methods=["GET"])
        except RouteExists:
            pass

        try:
            app.add_route(post_index_event, "/threads/<user_sub>/index-event", methods=["POST"])
        except RouteExists:
            pass

        return app

    core_run.configure_app = configure_app_with_thread_routes


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
    _install_version_route()
    _install_thread_index_routes()
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