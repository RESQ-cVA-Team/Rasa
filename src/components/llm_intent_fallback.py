# pyright: reportMissingTypeStubs=false, reportMissingModuleSource=false, reportUntypedClassDecorator=false, reportUntypedBaseClass=false
"""LLM-backed intent reclassification, triggered only when Rasa's own
classifier has already given up.

Runs strictly after FallbackClassifier in the pipeline and only does
anything when FallbackClassifier actually fired (i.e. this is genuinely a
fallback, not a second opinion on every message). The LLM is asked to pick
exactly one label from the real, domain-derived intent list -- enriched with
a few real NLU training examples per intent, sourced from whichever locale
this specific deployment was actually built for (see locale_detection.py).
Any response that isn't an exact match to the closed intent set is discarded
and the original nlu_fallback prediction is left untouched. That closed-set
validation is what keeps this safe against prompt injection -- the model's
raw text is never trusted or interpreted, only compared against a fixed
allow-list.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Text

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers.fallback_classifier import is_fallback_classifier_prediction
from rasa.shared.nlu.constants import INTENT, INTENT_NAME_KEY, INTENT_RANKING_KEY, PREDICTED_CONFIDENCE_KEY
from rasa.shared.nlu.training_data.message import Message

from src.components.intent_matching import bucket_examples, build_intent_list_block, match_intent_strict
from src.components.layered_importer import OverlayImporter
from src.components.llm_client import ChatMessage, OpenAICompatibleLLMClient, build_default_client
from src.components.locale_detection import detect_locale_overlay_domain
from src.components.prompt_template import render_system_prompt

logger = logging.getLogger(__name__)

_EXCLUDED_INTENTS = {"nlu_fallback"}

# Editable via config.yml's system_prompt_template (and, in CVaLab's pipeline
# editor, a text field on this component) -- see render_system_prompt.
# {intent_list} is required: the whole point of this component is picking
# from the real intent set, so a template that drops it gets rejected back
# to this default rather than silently sending an unconstrained prompt.
# {fallback_label} is optional -- match_intent_strict() is what actually
# keeps this component safe (a raw response has to byte-for-byte match a
# real intent regardless of what the prompt says), so an edited template
# omitting it is a quality choice, not a safety one.
_DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are an intent classifier. Read the user's message and reply with \
exactly one label from this list, and nothing else -- no punctuation, no explanation:

{intent_list}

If none of the labels genuinely fit, reply with: {fallback_label}"""

_REQUIRED_PROMPT_PLACEHOLDERS = ["{intent_list}"]

_DEFAULT_BASE_DOMAIN = ["src/core/domain"]
_DEFAULT_EXAMPLES_PER_INTENT = 2


def _load_domain_intents(base_domain: List[str], overlay_domain: List[str]) -> List[str]:
    """Reuse OverlayImporter (the same code that builds the real domain at
    train time) rather than re-deriving the intent list from scratch.

    Deliberately does NOT rely on the OVERLAY_BASE_DOMAIN/OVERLAY_DOMAIN env
    vars that the training scripts export: those only exist for the lifetime
    of the separate `bash scripts/layer_rasa_lang.sh ...` subprocess that
    builds the model, not in the actual serving process afterwards (verified
    against the real container: `docker exec rasa env | grep OVERLAY` finds
    nothing). Defaults to the same base domain path already hardcoded in
    config.yml's own `importers:` section, so this works out of the box in
    real deployment; OverlayImporter will still honor those env vars on top
    of these defaults if a caller does set them (e.g. manual testing)."""
    try:
        domain = OverlayImporter(base_domain=base_domain, overlay_domain=overlay_domain).get_domain()
        intents = getattr(domain, "intents", None) or []
        return sorted({str(i) for i in intents if str(i) not in _EXCLUDED_INTENTS})
    except Exception:
        logger.warning("Could not load domain intents for LLM fallback; component will be a no-op", exc_info=True)
        return []


def _load_intent_examples(
    base_domain: List[str],
    overlay_domain: List[str],
    examples_per_intent: int,
) -> Dict[str, List[str]]:
    """Real NLU training example utterances, grouped by intent, from
    whichever locale this deployment actually serves (overlay_domain --
    see locale_detection.py). Filtering/capping/dedup logic lives in
    bucket_examples() (intent_matching.py) so it's testable without rasa;
    this function is just the rasa-dependent data-loading half."""
    if examples_per_intent <= 0:
        return {}

    try:
        nlu_data = OverlayImporter(base_domain=base_domain, overlay_domain=overlay_domain).get_nlu_data()
    except Exception:
        logger.warning("Could not load NLU examples for LLM fallback; continuing without them", exc_info=True)
        return {}

    raw_examples = [
        (example.get("intent"), example.get("text"))
        for example in getattr(nlu_data, "training_examples", [])
        if isinstance(example.get("intent"), str) and isinstance(example.get("text"), str)
    ]
    return bucket_examples(raw_examples, examples_per_intent)


@DefaultV1Recipe.register(DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False)
class LLMIntentFallback(GraphComponent):
    """If FallbackClassifier gave up on a message, ask an LLM to pick one of
    the real intents before accepting defeat. Never runs on messages Rasa was
    already confident about."""

    def __init__(self, config: Dict[Text, Any], llm_client: Optional[OpenAICompatibleLLMClient] = None) -> None:
        self._config = config or {}
        self._enabled = bool(self._config.get("enabled", True))
        self._llm_confidence = float(self._config.get("llm_confidence", 0.5))
        self._max_tokens = int(self._config.get("max_tokens", 20))
        self._debug = bool(self._config.get("debug_logging", False))
        examples_per_intent = int(self._config.get("examples_per_intent", _DEFAULT_EXAMPLES_PER_INTENT))

        timeout_seconds = float(self._config.get("timeout_seconds", 8.0))
        self._client = llm_client if llm_client is not None else build_default_client(timeout_seconds)

        base_domain = self._config.get("base_domain", _DEFAULT_BASE_DOMAIN)
        overlay_domain = self._config.get("overlay_domain")
        if overlay_domain is None:
            overlay_domain = detect_locale_overlay_domain()

        self._intents: List[str] = _load_domain_intents(base_domain, overlay_domain)
        examples = _load_intent_examples(base_domain, overlay_domain, examples_per_intent) if self._intents else {}
        template = self._config.get("system_prompt_template", _DEFAULT_SYSTEM_PROMPT_TEMPLATE)
        self._system_prompt, prompt_warning = render_system_prompt(
            template,
            _DEFAULT_SYSTEM_PROMPT_TEMPLATE,
            _REQUIRED_PROMPT_PLACEHOLDERS,
            intent_list=build_intent_list_block(self._intents, examples),
            fallback_label="none",
        )
        if prompt_warning:
            logger.warning(f"LLMIntentFallback: {prompt_warning}")

        if self._enabled and not self._client.enabled:
            logger.info("LLMIntentFallback: LLM client not configured, component will be a no-op")
        if self._enabled and not self._intents:
            logger.info("LLMIntentFallback: no domain intents loaded, component will be a no-op")
        elif self._debug:
            example_count = sum(len(v) for v in examples.values())
            logger.info(
                f"LLMIntentFallback: loaded {len(self._intents)} intents, "
                f"{example_count} example utterances (overlay_domain={overlay_domain or 'none/base-only'})"
            )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "LLMIntentFallback":
        return cls(config)

    def process(self, messages: List[Message]) -> List[Message]:  # type: ignore[override]
        if not self._enabled or not self._client.enabled or not self._intents:
            return messages

        for message in messages:
            if not is_fallback_classifier_prediction(message.data):
                continue

            text = message.get("text") or ""
            if not text.strip():
                continue

            llm_intent = self._classify(text)
            if llm_intent is None:
                if self._debug:
                    logger.info(f"LLMIntentFallback: no usable answer for {text!r}, keeping nlu_fallback")
                continue

            if self._debug:
                logger.info(f"LLMIntentFallback: reclassified {text!r} -> {llm_intent!r}")

            new_prediction = {INTENT_NAME_KEY: llm_intent, PREDICTED_CONFIDENCE_KEY: self._llm_confidence, "llm_fallback": True}
            message.data[INTENT] = new_prediction
            message.data.setdefault(INTENT_RANKING_KEY, [])
            message.data[INTENT_RANKING_KEY].insert(0, new_prediction)

        return messages

    def _classify(self, text: str) -> Optional[str]:
        messages: List[ChatMessage] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": text},
        ]

        raw = self._client.complete(messages, max_tokens=self._max_tokens, temperature=0.0)
        if raw is None:
            return None

        return match_intent_strict(raw, self._intents)
