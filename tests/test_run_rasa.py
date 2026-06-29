import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def load_run_rasa_module():
    rasa_module = types.ModuleType("rasa")
    rasa_module.__version__ = "test"

    rasa_main_module = types.ModuleType("rasa.__main__")
    rasa_main_module.main = lambda: None

    rasa_core_module = types.ModuleType("rasa.core")
    rasa_core_run_module = types.ModuleType("rasa.core.run")
    rasa_core_run_module.configure_app = lambda *args, **kwargs: None

    sanic_module = types.ModuleType("sanic")
    sanic_response_module = types.ModuleType("sanic.response")
    sanic_response_module.json = lambda payload, status=200: {"payload": payload, "status": status}
    sanic_module.response = sanic_response_module

    sanic_routing_module = types.ModuleType("sanic_routing")
    sanic_routing_exceptions_module = types.ModuleType("sanic_routing.exceptions")

    class RouteExists(Exception):
        pass

    sanic_routing_exceptions_module.RouteExists = RouteExists

    module_map = {
        "rasa": rasa_module,
        "rasa.__main__": rasa_main_module,
        "rasa.core": rasa_core_module,
        "rasa.core.run": rasa_core_run_module,
        "sanic": sanic_module,
        "sanic.response": sanic_response_module,
        "sanic_routing": sanic_routing_module,
        "sanic_routing.exceptions": sanic_routing_exceptions_module,
    }

    module_path = Path(__file__).resolve().parents[1] / "src/run_rasa.py"
    spec = importlib.util.spec_from_file_location("run_rasa_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load run_rasa from {module_path}")

    with mock.patch.dict(sys.modules, module_map, clear=False):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    return module


run_rasa = load_run_rasa_module()


class RunRasaTests(unittest.TestCase):
    def test_read_env_trims_values_and_normalizes_empty_strings(self) -> None:
        with mock.patch.dict(sys.modules["os"].environ, {"RASA_TEST": "  value  "}, clear=False):
            self.assertEqual(run_rasa._read_env("RASA_TEST"), "value")

        with mock.patch.dict(sys.modules["os"].environ, {"RASA_TEST": "   "}, clear=False):
            self.assertIsNone(run_rasa._read_env("RASA_TEST"))

    def test_env_flag_parses_truthy_and_falsy_values(self) -> None:
        with mock.patch.dict(sys.modules["os"].environ, {"RASA_FLAG": "yes"}, clear=False):
            self.assertTrue(run_rasa._env_flag("RASA_FLAG", default=False))

        with mock.patch.dict(sys.modules["os"].environ, {"RASA_FLAG": "no"}, clear=False):
            self.assertFalse(run_rasa._env_flag("RASA_FLAG", default=True))

    def test_resolve_endpoints_file_uses_explicit_file_or_backend_preset(self) -> None:
        with mock.patch.dict(sys.modules["os"].environ, {"RASA_ENDPOINTS_FILE": "custom.yml"}, clear=False):
            self.assertEqual(run_rasa._resolve_endpoints_file(), "custom.yml")

        with mock.patch.dict(sys.modules["os"].environ, {"RASA_ENDPOINTS_FILE": "", "RASA_TRACKER_STORE_BACKEND": "memory"}, clear=False):
            self.assertEqual(run_rasa._resolve_endpoints_file(), "src/core/endpoints.memory.yml")

    def test_resolve_auth_token_requires_token_when_enabled(self) -> None:
        with mock.patch.dict(
            sys.modules["os"].environ,
            {"RASA_REQUIRE_AUTH_TOKEN": "true", "RASA_AUTH_TOKEN": ""},
            clear=False,
        ):
            with self.assertRaises(RuntimeError):
                run_rasa._resolve_auth_token()

        with mock.patch.dict(
            sys.modules["os"].environ,
            {"RASA_REQUIRE_AUTH_TOKEN": "false", "RASA_AUTH_TOKEN": ""},
            clear=False,
        ):
            self.assertEqual(run_rasa._resolve_auth_token(), "")

    def test_resolve_cors_accepts_only_bare_http_or_https_origins(self) -> None:
        with mock.patch.dict(sys.modules["os"].environ, {"RASA_CORS": "https://example.com"}, clear=False):
            self.assertEqual(run_rasa._resolve_cors(), "https://example.com")

        with mock.patch.dict(sys.modules["os"].environ, {"RASA_CORS": "https://example.com/path"}, clear=False):
            with self.assertRaises(RuntimeError):
                run_rasa._resolve_cors()

        with mock.patch.dict(sys.modules["os"].environ, {"RASA_CORS": "*"}, clear=False):
            with self.assertRaises(RuntimeError):
                run_rasa._resolve_cors()


if __name__ == "__main__":
    unittest.main()