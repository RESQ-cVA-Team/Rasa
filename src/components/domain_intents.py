# pyright: reportMissingTypeStubs=false, reportMissingModuleSource=false
"""Loads the real intent/entity/example set an LLM-backed NLU component needs
to ground its prompt against, from the same domain OverlayImporter builds at
train time -- rasa-dependent, so kept separate from the pure prompt/response
helpers in intent_matching.py.

Shared by llm_intent_fallback.py and llm_nlu_classifier.py so both prompts
are built from one real, always-current source instead of two copies that
could drift.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.components import ssot_yaml
from src.components.intent_matching import bucket_examples
from src.components.layered_importer import OverlayImporter

logger = logging.getLogger(__name__)

_EXCLUDED_INTENTS = {"nlu_fallback"}
DEFAULT_BASE_DOMAIN = ["src/core/domain"]

# Same default mapping SSOTCanonicalizer uses (src/components/ssot_canonicalizer.py)
# -- duplicated rather than imported from there since that module's default is an
# implementation detail of its own config, not a shared constant today.
_ENTITY_SSOT_FILES: Dict[str, str] = {
    "metric": "MetricType.yml",
    "chart_type": "ChartType.yml",
    "group_by": "GroupByType.yml",
    "operator_type": "OperatorType.yml",
    "sex": "SexType.yml",
    "stroke_type": "StrokeType.yml",
    "boolean_type": "BooleanType.yml",
    "statistical_test_type": "StatisticalTestType.yml",
}


def load_domain_intents(base_domain: List[str], overlay_domain: List[str]) -> List[str]:
    """Deliberately does NOT rely on the OVERLAY_BASE_DOMAIN/OVERLAY_DOMAIN env
    vars the training scripts export: those only exist for the lifetime of the
    separate `bash scripts/layer_rasa_lang.sh ...` subprocess that builds the
    model, not in the actual serving process afterwards (verified against the
    real container: `docker exec rasa env | grep OVERLAY` finds nothing).
    Defaults to the same base domain path already hardcoded in config.yml's
    own `importers:` section, so this works out of the box in real
    deployment; OverlayImporter will still honor those env vars on top of
    these defaults if a caller does set them (e.g. manual testing)."""
    try:
        domain = OverlayImporter(base_domain=base_domain, overlay_domain=overlay_domain).get_domain()
        intents = getattr(domain, "intents", None) or []
        return sorted({str(i) for i in intents if str(i) not in _EXCLUDED_INTENTS})
    except Exception:
        logger.warning("Could not load domain intents; component will be a no-op", exc_info=True)
        return []


def load_domain_entities(base_domain: List[str], overlay_domain: List[str]) -> List[str]:
    try:
        domain = OverlayImporter(base_domain=base_domain, overlay_domain=overlay_domain).get_domain()
        entities = getattr(domain, "entities", None) or []
        return sorted({str(e) for e in entities})
    except Exception:
        logger.warning("Could not load domain entities; component will be a no-op", exc_info=True)
        return []


def load_intent_examples(
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
        logger.warning("Could not load NLU examples; continuing without them", exc_info=True)
        return {}

    raw_examples = [
        (example.get("intent"), example.get("text"))
        for example in getattr(nlu_data, "training_examples", [])
        if isinstance(example.get("intent"), str) and isinstance(example.get("text"), str)
    ]
    return bucket_examples(raw_examples, examples_per_intent)


def load_entity_examples(
    entity_types: List[str],
    ssot_dir: str,
    locale: str,
    examples_per_entity: int,
) -> Dict[str, List[str]]:
    """A handful of real SSOT synonym strings per entity type, for an LLM
    prompt to see what an actual mention looks like.

    Only entity types this repo actually grounds against SSOT (the same set
    SSOTCanonicalizer maps, see _ENTITY_SSOT_FILES) get examples; the rest
    (age, nihss, hospital_name, ...) are generic enough that the type name
    alone is enough context, confirmed in testing -- SSOT-backed entities
    are exactly the ones with jargon-heavy, non-obvious real-world phrasing
    an LLM has no way to guess from the bare type name (e.g. "metric").
    """
    if examples_per_entity <= 0:
        return {}

    base = Path(ssot_dir)
    examples: Dict[str, List[str]] = {}
    for entity_type in entity_types:
        filename = _ENTITY_SSOT_FILES.get(entity_type)
        if not filename:
            continue
        path = base / filename
        if not path.exists():
            continue
        try:
            items = ssot_yaml.load_ssot_items(path)
        except Exception:
            logger.warning(f"Could not load SSOT examples from {path}", exc_info=True)
            continue

        collected: List[str] = []
        for item in items:
            synonyms = ssot_yaml.locale_synonyms(item, locale) or ssot_yaml.all_synonyms(item)
            if synonyms:
                collected.append(synonyms[0])
            if len(collected) >= examples_per_entity:
                break
        if collected:
            examples[entity_type] = collected

    return examples
