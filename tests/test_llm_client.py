import unittest
from unittest import mock

from src.components.llm_client import LLMConfigError, OpenAICompatibleLLMClient, _resolve_base_url, _resolve_api_key


class ResolveBaseUrlTests(unittest.TestCase):
    def test_openai_defaults_to_public_api(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_resolve_base_url("openai"), "https://api.openai.com/v1")

    def test_openai_respects_explicit_base_url(self) -> None:
        with mock.patch.dict("os.environ", {"LLM_BASE_URL": "https://azure.example.com/v1"}, clear=True):
            self.assertEqual(_resolve_base_url("openai"), "https://azure.example.com/v1")

    def test_ollama_defaults_and_appends_v1(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_resolve_base_url("ollama"), "http://ollama:11434/v1")

    def test_ollama_respects_explicit_base_url(self) -> None:
        with mock.patch.dict("os.environ", {"LLM_BASE_URL": "http://localhost:11434/"}, clear=True):
            self.assertEqual(_resolve_base_url("ollama"), "http://localhost:11434/v1")

    def test_openai_compatible_requires_base_url(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(LLMConfigError):
                _resolve_base_url("openai-compatible")

    def test_vllm_and_sglang_require_base_url(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(LLMConfigError):
                _resolve_base_url("vllm")
            with self.assertRaises(LLMConfigError):
                _resolve_base_url("sglang")


class ResolveApiKeyTests(unittest.TestCase):
    def test_ollama_does_not_require_a_key(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(_resolve_api_key("ollama"))

    def test_openai_requires_a_key(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(LLMConfigError):
                _resolve_api_key("openai")

    def test_falls_back_to_openai_api_key_env_var(self) -> None:
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            self.assertEqual(_resolve_api_key("openai-compatible"), "sk-test")


class ClientConfigurationTests(unittest.TestCase):
    def test_disabled_when_provider_or_model_missing(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            client = OpenAICompatibleLLMClient()
            self.assertFalse(client.enabled)
            self.assertIsNone(client.complete([{"role": "user", "content": "hi"}]))

    def test_disabled_when_misconfigured(self) -> None:
        # openai-compatible with no LLM_BASE_URL -- should disable, not raise.
        env = {"LLM_PROVIDER": "openai-compatible", "LLM_MODEL": "some-model"}
        with mock.patch.dict("os.environ", env, clear=True):
            client = OpenAICompatibleLLMClient()
            self.assertFalse(client.enabled)

    def test_enabled_with_valid_config(self) -> None:
        env = {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-test",
            "LLM_API_KEY": "sk-test",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            client = OpenAICompatibleLLMClient()
            self.assertTrue(client.enabled)


class ExtraKwargsTests(unittest.TestCase):
    ENV = {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-test", "LLM_API_KEY": "sk-test"}

    def _sent_payload(self, kwargs_json: str) -> dict:
        response = mock.Mock()
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with mock.patch.dict("os.environ", {**self.ENV, "LLM_KWARGS_JSON": kwargs_json}, clear=True):
            client = OpenAICompatibleLLMClient()
            with mock.patch("src.components.llm_client.requests.post", return_value=response) as post:
                self.assertEqual(client.complete([{"role": "user", "content": "hi"}], max_tokens=20), "ok")
        return post.call_args.kwargs["json"]

    def test_sends_llm_kwargs_json_with_the_request(self) -> None:
        self.assertEqual(self._sent_payload('{"reasoning_effort": "none"}')["reasoning_effort"], "none")

    def test_the_call_own_values_win_over_llm_kwargs_json(self) -> None:
        payload = self._sent_payload('{"max_tokens": 4000, "model": "other"}')
        self.assertEqual(payload["max_tokens"], 20)
        self.assertEqual(payload["model"], "gpt-test")

    def test_disabled_when_llm_kwargs_json_is_invalid(self) -> None:
        for raw in ("not json", "[1]"):
            with mock.patch.dict("os.environ", {**self.ENV, "LLM_KWARGS_JSON": raw}, clear=True):
                self.assertFalse(OpenAICompatibleLLMClient().enabled)


if __name__ == "__main__":
    unittest.main()
