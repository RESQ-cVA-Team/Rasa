# pyright: reportMissingTypeStubs=false, reportMissingModuleSource=false, reportUntypedClassDecorator=false, reportUntypedBaseClass=false
"""EXPERIMENTAL: an LLM-only replacement for DIETClassifier + RegexEntityExtractor
-- one call classifies the intent and extracts every entity, instead of a
trained neural classifier plus a separate lookup-table extractor.

Not wired into the real config.yml; see the branch this lives on. Slots into
the exact same pipeline position and produces the exact same message.data
shape (INTENT / INTENT_RANKING_KEY / ENTITIES) those two components would
have, so everything downstream -- EntitySynonymMapper, SSOTCanonicalizer,
EntityConsolidator, FallbackClassifier, LLMIntentFallback, the policies --
runs completely unmodified. A message this component can't get a usable
answer for is left unclassified, same as a DIETClassifier prediction below
threshold: FallbackClassifier's own threshold check (message.data[INTENT] is
never set) handles it exactly as it already does today.

Not trainable: `is_trainable=False`, same as LLMIntentFallback. There's
nothing to fit -- the prompt is built once at construction time from the
real domain (intents, entities, example utterances), and every message just
gets one inference call.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Text

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    EXTRACTOR,
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
)
from rasa.shared.nlu.training_data.message import Message

from src.components.domain_intents import (
    DEFAULT_BASE_DOMAIN,
    load_domain_entities,
    load_domain_intents,
    load_entity_examples,
    load_intent_examples,
)
from src.components.intent_matching import build_intent_list_block
from src.components.llm_client import ChatMessage, OpenAICompatibleLLMClient, build_default_client
from src.components.llm_nlu_parsing import (
    DEFAULT_INSTRUCTIONS_TEMPLATE,
    REQUIRED_PROMPT_PLACEHOLDERS,
    build_entity_type_block,
    build_system_prompt,
    parse_llm_nlu_response,
)
from src.components.locale_detection import detect_locale_overlay_domain
from src.components.prompt_template import render_system_prompt

logger = logging.getLogger(__name__)

_DEFAULT_EXAMPLES_PER_INTENT = 3
_DEFAULT_EXAMPLES_PER_ENTITY = 5
_LLM_EXTRACTOR_NAME = "LLMNluClassifier"


@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR],
    is_trainable=False,
)
class LLMNluClassifier(GraphComponent):
    def __init__(self, config: Dict[Text, Any], llm_client: Optional[OpenAICompatibleLLMClient] = None) -> None:
        self._config = config or {}
        self._max_tokens = int(self._config.get("max_tokens", 400))
        self._debug = bool(self._config.get("debug_logging", False))
        self._fallback_confidence = float(self._config.get("fallback_confidence", 0.1))
        self._default_confidence = float(self._config.get("default_confidence", 0.85))
        examples_per_intent = int(self._config.get("examples_per_intent", _DEFAULT_EXAMPLES_PER_INTENT))
        examples_per_entity = int(self._config.get("examples_per_entity", _DEFAULT_EXAMPLES_PER_ENTITY))
        ssot_dir = self._config.get("ssot_dir", "src/shared/SSOT")

        timeout_seconds = float(self._config.get("timeout_seconds", 15.0))
        self._client = llm_client if llm_client is not None else build_default_client(timeout_seconds)

        base_domain = self._config.get("base_domain", DEFAULT_BASE_DOMAIN)
        overlay_domain = self._config.get("overlay_domain")
        if overlay_domain is None:
            overlay_domain = detect_locale_overlay_domain()

        self._intents: List[str] = load_domain_intents(base_domain, overlay_domain)
        self._entity_types: List[str] = load_domain_entities(base_domain, overlay_domain)
        intent_examples = load_intent_examples(base_domain, overlay_domain, examples_per_intent) if self._intents else {}
        entity_examples = (
            load_entity_examples(self._entity_types, ssot_dir, "en", examples_per_entity) if self._entity_types else {}
        )
        template = self._config.get("system_prompt_template", DEFAULT_INSTRUCTIONS_TEMPLATE)
        instructions, prompt_warning = render_system_prompt(
            template,
            DEFAULT_INSTRUCTIONS_TEMPLATE,
            REQUIRED_PROMPT_PLACEHOLDERS,
            intent_list=build_intent_list_block(self._intents, intent_examples),
            entity_type_list=build_entity_type_block(self._entity_types, entity_examples),
        )
        if prompt_warning:
            logger.warning(f"LLMNluClassifier: {prompt_warning}")
        self._system_prompt = build_system_prompt(instructions)

        if not self._client.enabled:
            logger.warning("LLMNluClassifier: LLM client not configured, every message will go unclassified")
        if not self._intents:
            logger.warning("LLMNluClassifier: no domain intents loaded, component will be a no-op")
        elif self._debug:
            logger.info(
                f"LLMNluClassifier: loaded {len(self._intents)} intents, "
                f"{len(self._entity_types)} entity types (overlay_domain={overlay_domain or 'none/base-only'})"
            )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "LLMNluClassifier":
        return cls(config)

    def process(self, messages: List[Message]) -> List[Message]:  # type: ignore[override]
        if not self._client.enabled or not self._intents:
            return messages

        for message in messages:
            text = message.get("text") or ""
            if not text.strip():
                continue

            raw = self._client.complete(
                [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
            parsed = parse_llm_nlu_response(
                raw,
                self._intents,
                self._entity_types,
                text,
                fallback_confidence=self._fallback_confidence,
                default_confidence=self._default_confidence,
            )
            if parsed is None:
                if self._debug:
                    logger.info(f"LLMNluClassifier: no usable answer for {text!r}, leaving unclassified")
                continue

            if self._debug:
                logger.info(
                    f"LLMNluClassifier: {text!r} -> intent={parsed.intent!r} "
                    f"confidence={parsed.confidence:.2f} entities={parsed.entities}"
                )

            # message.set(..., add_to_output=True), not raw message.data[...] =
            # assignment: the final parse result only includes keys registered
            # in Message.output_properties (see MessageProcessor's
            # as_dict(only_output_properties=True)). DIETClassifier registers
            # "intent" itself in the stock pipeline, which is why
            # FallbackClassifier/LLMIntentFallback's own raw dict mutations of
            # message.data[INTENT] work today -- the key's already registered
            # by the time they run. With DIETClassifier removed, nothing
            # registers it unless this component does.
            prediction = {INTENT_NAME_KEY: parsed.intent, PREDICTED_CONFIDENCE_KEY: parsed.confidence}
            message.set(INTENT, prediction, add_to_output=True)
            message.set(INTENT_RANKING_KEY, [prediction], add_to_output=True)

            if parsed.entities:
                entity_dicts: List[Dict[str, Any]] = []
                for entity in parsed.entities:
                    entity_dict: Dict[str, Any] = {
                        ENTITY_ATTRIBUTE_TYPE: entity.entity,
                        ENTITY_ATTRIBUTE_VALUE: entity.value,
                        ENTITY_ATTRIBUTE_START: entity.start,
                        ENTITY_ATTRIBUTE_END: entity.end,
                        EXTRACTOR: _LLM_EXTRACTOR_NAME,
                    }
                    if entity.role:
                        entity_dict[ENTITY_ATTRIBUTE_ROLE] = entity.role
                    entity_dicts.append(entity_dict)
                message.set(ENTITIES, message.get(ENTITIES, []) + entity_dicts, add_to_output=True)

        return messages
