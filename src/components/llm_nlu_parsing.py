"""Pure, rasa-independent helpers for the LLM NLU classifier's prompt/response
handling -- parsing the model's JSON, validating it against the real closed
sets (intents, entity types), and locating each entity's character span in
the original text. Kept separate from llm_nlu_classifier.py so all of it is
testable without importing rasa at all, same reasoning as intent_matching.py.

Division of labor with SSOTCanonicalizer (which runs later in the pipeline,
unchanged): this module's job is "did the model name a real intent, a real
entity type, and text that actually appears in the message" -- structural
validation only. Whether an entity's extracted text maps to a real SSOT
canonical value is SSOTCanonicalizer's job, exactly as it already is for
RegexEntityExtractor's raw lookup-table matches. An entity this module
passes through is a candidate, not yet a grounded value.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ParsedEntity:
    entity: str
    value: str
    start: int
    end: int
    role: Optional[str] = None


@dataclass(frozen=True)
class ParsedNlu:
    intent: str
    confidence: float
    entities: List[ParsedEntity] = field(default_factory=list)


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Models routinely wrap JSON in prose or a fenced code block even when
    told not to. Take the outermost {...} span rather than requiring the
    whole response to be bare JSON -- still fails closed (returns None) if
    that span isn't valid JSON, never guesses."""
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def find_entity_span(text: str, entity_text: str) -> Optional[tuple[int, int]]:
    """Locates `entity_text` inside the original message, case-insensitively.
    The model is asked for the verbatim substring, not a paraphrase, so a
    normal string search is enough -- no fuzzy matching, which would risk
    anchoring an entity to the wrong span. Returns None (entity dropped by
    the caller) if the model's text doesn't actually appear in the message;
    that's the anti-hallucination check for spans, same spirit as
    match_intent_strict's closed-set check for intents."""
    if not entity_text or not entity_text.strip():
        return None
    idx = text.lower().find(entity_text.lower())
    if idx == -1:
        return None
    return idx, idx + len(entity_text)


def parse_llm_nlu_response(
    raw_response: Optional[str],
    valid_intents: List[str],
    valid_entity_types: List[str],
    text: str,
    fallback_confidence: float = 0.1,
    default_confidence: float = 0.85,
) -> Optional[ParsedNlu]:
    """Turns the model's raw completion into a ParsedNlu, or None if the
    response isn't usable at all (caller should leave the message
    unclassified so FallbackClassifier's own threshold handles it, same as
    a DIETClassifier prediction below threshold would).

    Validation, all closed-set / structural -- nothing here trusts the
    model's text as anything other than data to check against real values:
    - intent must exactly match one real domain intent (case-insensitive),
      or the reserved value "none" for "doesn't fit any of them" -- either
      way this function always returns a ParsedNlu (never None) once the
      JSON itself parses, so a deliberate "none" still reaches
      FallbackClassifier with a real low-confidence prediction rather than
      silently falling through as an unparsed response would.
    - each entity's type must be one of the real domain entities; unknown
      types are dropped, not kept as free-form data.
    - each entity's text must be found verbatim in the message (see
      find_entity_span); entities that aren't are dropped.
    - confidence is clamped to [0, 1] regardless of what the model sent.
    """
    parsed = _extract_json_object(raw_response or "")
    if parsed is None:
        return None

    intent_raw = parsed.get("intent")
    if not isinstance(intent_raw, str) or not intent_raw.strip():
        return None

    intent_by_lower = {i.lower(): i for i in valid_intents}
    candidate = intent_raw.strip().lower()
    if candidate == "none":
        intent = "none"
        confidence = fallback_confidence
    elif candidate in intent_by_lower:
        intent = intent_by_lower[candidate]
        confidence_raw = parsed.get("confidence")
        confidence = _clamp_confidence(confidence_raw, default_confidence)
    else:
        return None

    entities: List[ParsedEntity] = []
    entity_types = set(valid_entity_types)
    raw_entities = parsed.get("entities")
    if isinstance(raw_entities, list):
        for raw_entity in raw_entities:
            parsed_entity = _parse_entity(raw_entity, entity_types, text)
            if parsed_entity is not None:
                entities.append(parsed_entity)

    return ParsedNlu(intent=intent, confidence=confidence, entities=entities)


def _parse_entity(raw_entity: Any, valid_entity_types: set[str], text: str) -> Optional[ParsedEntity]:
    if not isinstance(raw_entity, dict):
        return None

    entity_type = raw_entity.get("entity")
    entity_text = raw_entity.get("text")
    if not isinstance(entity_type, str) or entity_type not in valid_entity_types:
        return None
    if not isinstance(entity_text, str):
        return None

    span = find_entity_span(text, entity_text)
    if span is None:
        return None
    start, end = span

    # Despite the schema hint saying "optional", the model sometimes fills it
    # in literally with "none"/"null"/"n/a" rather than omitting the key --
    # those aren't real role values.
    role_raw = raw_entity.get("role")
    role = role_raw.strip() if isinstance(role_raw, str) and role_raw.strip() else None
    if role is not None and role.lower() in {"none", "null", "n/a", "na"}:
        role = None

    # value = the verbatim text, same as RegexEntityExtractor's raw
    # lookup-table match -- SSOTCanonicalizer (downstream, unchanged) maps
    # it to a real canonical code from there, exactly as it already does.
    return ParsedEntity(entity=entity_type, value=text[start:end], start=start, end=end, role=role)


def _clamp_confidence(value: Any, default: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return default
    return max(0.0, min(1.0, float(value)))


def build_entity_type_block(entity_types: List[str], examples: Optional[Dict[str, List[str]]] = None) -> str:
    """Mirrors build_intent_list_block's shape (intent_matching.py): a type
    name per line, with a couple of real example values indented under it
    where available. Confirmed in testing that this matters a lot for
    jargon-heavy types like "metric" -- the bare type name alone isn't
    enough for the model to recognize domain-specific phrasing it has no
    other way to know about; more self-evident types (sex, chart_type) work
    fine without any."""
    examples = examples or {}
    lines: List[str] = []
    for name in entity_types:
        lines.append(f"- {name}")
        for example_value in examples.get(name, []):
            lines.append(f'  e.g. "{example_value}"')
    return "\n".join(lines)


# Fixed, never editable via config: parse_llm_nlu_response() reads these
# exact JSON key names ("intent", "confidence", "entities", "entity", "text",
# "role"). Unlike LLMIntentFallback's prompt (where safety comes entirely
# from match_intent_strict()'s closed-set check on the raw response, with no
# coupling to prompt wording at all), an edit that changed these field names
# here wouldn't just degrade quality -- parsing would silently find nothing,
# since it looks up these exact keys. The editable part is DEFAULT_INSTRUCTIONS_TEMPLATE
# below; this stays code-owned.
_JSON_SCHEMA_HINT = """Reply with a single JSON object and nothing else -- no prose, no markdown fences:
{
  "intent": "<one label from the intent list, or \\"none\\" if nothing fits>",
  "confidence": <your confidence in that intent, 0.0 to 1.0>,
  "entities": [
    {"entity": "<one type from the entity list>", "text": "<verbatim substring from the message>", "role": "<optional, e.g. upper/lower for a numeric range>"}
  ]
}
Only include entities whose type is in the entity list below, and only when the value is a verbatim substring of the message -- never paraphrase or invent one."""

# Editable via config.yml's system_prompt_template (and, in CVaLab's pipeline
# editor, a text field on this component) -- see render_system_prompt.
# {intent_list} and {entity_type_list} are both required: dropping either
# means the model gets no grounding for that half of its job, so a template
# missing one gets rejected back to this default rather than silently
# running unconstrained.
DEFAULT_INSTRUCTIONS_TEMPLATE = """You are an NLU engine: classify the user's intent and extract entities.

Valid intents:
{intent_list}

Valid entity types:
{entity_type_list}"""

REQUIRED_PROMPT_PLACEHOLDERS = ["{intent_list}", "{entity_type_list}"]


def build_system_prompt(instructions: str) -> str:
    """`instructions` is the (possibly admin-edited) rendered instructions
    block -- see render_system_prompt() in prompt_template.py. The JSON
    schema hint is always appended after it, unconditionally."""
    return f"{instructions}\n\n{_JSON_SCHEMA_HINT}"
