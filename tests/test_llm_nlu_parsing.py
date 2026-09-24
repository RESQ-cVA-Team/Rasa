import json
import unittest

from src.components.llm_nlu_parsing import (
    ParsedEntity,
    ParsedNlu,
    build_entity_type_block,
    build_system_prompt,
    find_entity_span,
    parse_llm_nlu_response,
)

_INTENTS = ["generate_visualization", "greet", "faq_chart_types"]
_ENTITY_TYPES = ["metric", "chart_type", "sex", "age"]


def _response(obj) -> str:
    return json.dumps(obj)


class FindEntitySpanTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertEqual(find_entity_span("show me door to needle", "door to needle"), (8, 22))

    def test_case_insensitive(self) -> None:
        self.assertEqual(find_entity_span("Show DOOR TO NEEDLE please", "door to needle"), (5, 19))

    def test_not_found_returns_none(self) -> None:
        self.assertIsNone(find_entity_span("show me door to needle", "onset to groin"))

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(find_entity_span("show me something", ""))
        self.assertIsNone(find_entity_span("show me something", "   "))


class ParseLlmNluResponseTests(unittest.TestCase):
    def test_valid_response_with_entity(self) -> None:
        raw = _response(
            {
                "intent": "generate_visualization",
                "confidence": 0.92,
                "entities": [{"entity": "metric", "text": "door to needle"}],
            }
        )
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "show me door to needle for men")
        self.assertEqual(
            result,
            ParsedNlu(
                intent="generate_visualization",
                confidence=0.92,
                entities=[ParsedEntity(entity="metric", value="door to needle", start=8, end=22)],
            ),
        )

    def test_intent_matched_case_insensitively_but_returns_real_casing(self) -> None:
        raw = _response({"intent": "GREET", "confidence": 0.9, "entities": []})
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "hi there")
        assert result is not None
        self.assertEqual(result.intent, "greet")

    def test_explicit_none_is_a_real_low_confidence_prediction_not_a_parse_failure(self) -> None:
        raw = _response({"intent": "none", "entities": []})
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "asdkjfh qwerty", fallback_confidence=0.1)
        self.assertEqual(result, ParsedNlu(intent="none", confidence=0.1, entities=[]))

    def test_intent_not_in_closed_set_is_rejected(self) -> None:
        raw = _response({"intent": "delete_all_users", "confidence": 0.99, "entities": []})
        self.assertIsNone(parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "hi"))

    def test_missing_intent_field_is_rejected(self) -> None:
        self.assertIsNone(parse_llm_nlu_response(_response({"entities": []}), _INTENTS, _ENTITY_TYPES, "hi"))

    def test_malformed_json_is_rejected(self) -> None:
        self.assertIsNone(parse_llm_nlu_response("not json at all", _INTENTS, _ENTITY_TYPES, "hi"))
        self.assertIsNone(parse_llm_nlu_response("", _INTENTS, _ENTITY_TYPES, "hi"))
        self.assertIsNone(parse_llm_nlu_response(None, _INTENTS, _ENTITY_TYPES, "hi"))

    def test_json_wrapped_in_prose_or_fences_is_still_parsed(self) -> None:
        raw = 'Sure, here you go:\n```json\n{"intent": "greet", "confidence": 0.9, "entities": []}\n```'
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "hi")
        assert result is not None
        self.assertEqual(result.intent, "greet")

    def test_entity_with_unknown_type_is_dropped_but_intent_kept(self) -> None:
        raw = _response(
            {
                "intent": "greet",
                "confidence": 0.9,
                "entities": [{"entity": "made_up_type", "text": "hi"}],
            }
        )
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "hi there")
        assert result is not None
        self.assertEqual(result.entities, [])

    def test_entity_whose_text_is_not_in_the_message_is_dropped(self) -> None:
        # The anti-hallucination check: a real entity type, but text the
        # model invented rather than quoted from the actual message.
        raw = _response(
            {
                "intent": "generate_visualization",
                "confidence": 0.9,
                "entities": [{"entity": "metric", "text": "onset to groin time"}],
            }
        )
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "show me door to needle")
        assert result is not None
        self.assertEqual(result.entities, [])

    def test_entity_role_is_preserved_when_present(self) -> None:
        raw = _response(
            {
                "intent": "generate_visualization",
                "confidence": 0.9,
                "entities": [{"entity": "age", "text": "45", "role": "upper"}],
            }
        )
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "patients younger than 45")
        assert result is not None
        self.assertEqual(result.entities[0].role, "upper")

    def test_literal_none_role_string_is_treated_as_no_role(self) -> None:
        # The model sometimes fills the "optional" role field with the literal
        # string "none" rather than omitting it.
        for literal in ("none", "None", "null", "n/a", "NA"):
            raw = _response(
                {
                    "intent": "greet",
                    "confidence": 0.9,
                    "entities": [{"entity": "age", "text": "45", "role": literal}],
                }
            )
            result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "45", default_confidence=0.9)
            assert result is not None
            self.assertIsNone(result.entities[0].role, msg=literal)

    def test_entities_not_a_list_is_treated_as_no_entities(self) -> None:
        raw = _response({"intent": "greet", "confidence": 0.9, "entities": "door to needle"})
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "hi")
        assert result is not None
        self.assertEqual(result.entities, [])

    def test_confidence_clamped_to_valid_range(self) -> None:
        raw = _response({"intent": "greet", "confidence": 5.0, "entities": []})
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "hi")
        assert result is not None
        self.assertEqual(result.confidence, 1.0)

        raw = _response({"intent": "greet", "confidence": -3.0, "entities": []})
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "hi")
        assert result is not None
        self.assertEqual(result.confidence, 0.0)

    def test_non_numeric_confidence_falls_back_to_default(self) -> None:
        raw = _response({"intent": "greet", "confidence": "very sure", "entities": []})
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "hi", default_confidence=0.77)
        assert result is not None
        self.assertEqual(result.confidence, 0.77)

    def test_missing_confidence_falls_back_to_default(self) -> None:
        raw = _response({"intent": "greet", "entities": []})
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "hi", default_confidence=0.6)
        assert result is not None
        self.assertEqual(result.confidence, 0.6)

    def test_bool_confidence_is_not_treated_as_numeric(self) -> None:
        # bool is a subclass of int in Python; True/False would otherwise
        # silently pass as confidence 1.0/0.0.
        raw = _response({"intent": "greet", "confidence": True, "entities": []})
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "hi", default_confidence=0.42)
        assert result is not None
        self.assertEqual(result.confidence, 0.42)

    def test_multiple_entities_of_different_types(self) -> None:
        raw = _response(
            {
                "intent": "generate_visualization",
                "confidence": 0.9,
                "entities": [
                    {"entity": "metric", "text": "door to needle"},
                    {"entity": "sex", "text": "men"},
                ],
            }
        )
        result = parse_llm_nlu_response(raw, _INTENTS, _ENTITY_TYPES, "show door to needle for men")
        assert result is not None
        self.assertEqual(len(result.entities), 2)
        self.assertEqual({e.entity for e in result.entities}, {"metric", "sex"})


class PromptBuildingTests(unittest.TestCase):
    def test_entity_type_block_lists_each_type(self) -> None:
        block = build_entity_type_block(["metric", "sex"])
        self.assertEqual(block, "- metric\n- sex")

    def test_entity_type_block_includes_examples_where_given(self) -> None:
        block = build_entity_type_block(["metric", "sex"], {"metric": ["door to needle", "pre-stroke mrs"]})
        self.assertEqual(
            block,
            '- metric\n  e.g. "door to needle"\n  e.g. "pre-stroke mrs"\n- sex',
        )

    def test_system_prompt_appends_the_fixed_json_schema_after_the_instructions(self) -> None:
        prompt = build_system_prompt("Custom instructions.\n- greet\n- metric")
        self.assertIn("Custom instructions.\n- greet\n- metric", prompt)
        self.assertIn('"intent"', prompt)
        self.assertIn('"entities"', prompt)


if __name__ == "__main__":
    unittest.main()
