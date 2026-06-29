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