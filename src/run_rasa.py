import os
import sys
from typing import Optional

import rasa
import rasa.__main__ as rasa_main
import rasa.core.run as core_run
from sanic import response


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

        @app.get("/version")
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

        return app

    core_run.configure_app = configure_app_with_version


def main() -> None:
    _install_version_route()
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
                "src/core/endpoints.yml",
                "--request-timeout",
                os.getenv("RASA_REQUEST_TIMEOUT", "300"),
                "--response-timeout",
                os.getenv("RASA_RESPONSE_TIMEOUT", "300"),
            ]
        )
    rasa_main.main()


if __name__ == "__main__":
    main()