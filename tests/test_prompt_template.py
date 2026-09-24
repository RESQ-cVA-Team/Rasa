import unittest

from src.components.prompt_template import render_system_prompt

_DEFAULT = "You classify intents.\n\n{intent_list}\n\nReply with: {fallback_label}"


class RenderSystemPromptTests(unittest.TestCase):
    def test_valid_template_renders_with_no_warning(self) -> None:
        template = "Custom instructions.\n\n{intent_list}"
        rendered, warning = render_system_prompt(
            template, _DEFAULT, required_placeholders=["{intent_list}"], intent_list="- greet", fallback_label="none"
        )
        self.assertEqual(rendered, "Custom instructions.\n\n- greet")
        self.assertIsNone(warning)

    def test_default_template_always_valid(self) -> None:
        rendered, warning = render_system_prompt(
            _DEFAULT, _DEFAULT, required_placeholders=["{intent_list}"], intent_list="- greet", fallback_label="none"
        )
        self.assertIn("- greet", rendered)
        self.assertIsNone(warning)

    def test_missing_required_placeholder_falls_back_to_default(self) -> None:
        template = "Custom instructions with no intent list at all."
        rendered, warning = render_system_prompt(
            template, _DEFAULT, required_placeholders=["{intent_list}"], intent_list="- greet", fallback_label="none"
        )
        self.assertEqual(rendered, _DEFAULT.format(intent_list="- greet", fallback_label="none"))
        assert warning is not None
        self.assertIn("intent_list", warning)

    def test_multiple_required_placeholders_all_checked(self) -> None:
        template = "{intent_list} only, no fallback label here"
        rendered, warning = render_system_prompt(
            template,
            _DEFAULT,
            required_placeholders=["{intent_list}", "{fallback_label}"],
            intent_list="- greet",
            fallback_label="none",
        )
        self.assertEqual(rendered, _DEFAULT.format(intent_list="- greet", fallback_label="none"))
        assert warning is not None
        self.assertIn("fallback_label", warning)

    def test_template_referencing_unknown_placeholder_falls_back(self) -> None:
        # Has {intent_list} (passes the required check) but also references
        # something never supplied -- str.format() would raise KeyError.
        template = "{intent_list}\n{nonexistent_field}"
        rendered, warning = render_system_prompt(
            template, _DEFAULT, required_placeholders=["{intent_list}"], intent_list="- greet", fallback_label="none"
        )
        self.assertEqual(rendered, _DEFAULT.format(intent_list="- greet", fallback_label="none"))
        assert warning is not None
        self.assertIn("failed to render", warning)

    def test_empty_required_placeholders_means_any_template_is_valid(self) -> None:
        template = "Anything goes here, no grounding required."
        rendered, warning = render_system_prompt(template, _DEFAULT, required_placeholders=[])
        self.assertEqual(rendered, template)
        self.assertIsNone(warning)

    def test_default_template_used_verbatim_when_template_equals_default(self) -> None:
        rendered, warning = render_system_prompt(
            _DEFAULT, _DEFAULT, required_placeholders=["{intent_list}"], intent_list="- a\n- b", fallback_label="x"
        )
        self.assertEqual(rendered, "You classify intents.\n\n- a\n- b\n\nReply with: x")
        self.assertIsNone(warning)


if __name__ == "__main__":
    unittest.main()
