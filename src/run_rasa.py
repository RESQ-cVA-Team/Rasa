import os
import sys
import warnings
from typing import Optional


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
        "redis": "src/core/endpoints.yml",
        "sql": "src/core/endpoints.sql.yml",
        "mongo": "src/core/endpoints.mongo.yml",
        "dynamo": "src/core/endpoints.dynamo.yml",
    }
    return presets.get(backend, "src/core/endpoints.yml")


def main() -> None:
    _install_version_route()
    endpoints_file = _resolve_endpoints_file()
    # Docker runs this entrypoint without CLI args by default; in that case,
    # provide the same sensible defaults we used previously.
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
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
        )
    rasa_main.main()


if __name__ == "__main__":
    main()