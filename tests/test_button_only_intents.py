import unittest

from src.components.llm_intent_fallback import _load_domain_intents


class ButtonOnlyIntentTests(unittest.TestCase):
    def test_the_llm_fallback_never_offers_a_button_only_intent(self) -> None:
        intents = _load_domain_intents(["src/core/domain"], [])
        self.assertIn("generate_visualization", intents)
        self.assertNotIn("show_answer_as_chart", intents)
        self.assertNotIn("nlu_fallback", intents)


if __name__ == "__main__":
    unittest.main()
