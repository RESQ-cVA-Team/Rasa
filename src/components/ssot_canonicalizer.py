# pyright: reportMissingTypeStubs=false, reportMissingModuleSource=false, reportUntypedClassDecorator=false, reportUntypedBaseClass=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Text, cast

from rasa.engine.graph import GraphComponent  # type: ignore
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # type: ignore
from rasa.shared.nlu.training_data.message import Message  # type: ignore

from src.components import ssot_yaml

logger = logging.getLogger(__name__)


def _norm_text(text: str) -> str:
    # Conservative normalization: keep non-latin characters, but normalize spacing and common separators.
    s = text.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass(frozen=True)
class _SSOTIndex:
    canonicals: Set[str]
    by_synonym: Dict[str, str]

    def lookup(self, raw_value: str) -> Optional[str]:
        key = _norm_text(raw_value)
        if not key:
            return None
        return self.by_synonym.get(key)


def _load_ssot_index(path: Path) -> _SSOTIndex:
    """Loads a SSOT YAML file into a synonym->canonical index.

    For MetricType.yml, some items also include data_type: Enum with an Enum list; we include
    enum keys + their synonyms as valid synonyms for that canonical as well.
    """

    items = ssot_yaml.load_ssot_items(path)

    canonicals: Set[str] = set()
    by_synonym: Dict[str, str] = {}

    for item in items:
        canonical_any = item.get("canonical")
        if not canonical_any:
            continue
        canonical = str(canonical_any)
        canonicals.add(canonical)

        synonyms = ssot_yaml.all_synonyms(item)
        # Always accept the canonical itself.
        synonyms.append(canonical)

        # For Enum types, also accept enum keys and their synonyms.
        if str(item.get("data_type") or "").lower() == "enum":
            enum_items = item.get("Enum")
            if isinstance(enum_items, list):
                for e in cast(List[Any], enum_items):
                    if not isinstance(e, dict):
                        continue
                    key_any = cast(Dict[str, Any], e).get("key")
                    if key_any is not None:
                        synonyms.append(str(key_any))
                    synonyms.extend(
                        ssot_yaml.all_synonyms(cast(Dict[str, Any], e))
                    )

        for syn in synonyms:
            k = _norm_text(syn)
            if not k:
                continue
            # First-one-wins to avoid accidental churn if duplicates exist.
            by_synonym.setdefault(k, canonical)

    return _SSOTIndex(canonicals=canonicals, by_synonym=by_synonym)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False
)
class SSOTCanonicalizer(GraphComponent):
    """Normalizes SSOT-backed entity values to canonical codes.

    Behavior:
      - For configured entity types, if extracted value matches a SSOT synonym, rewrite `value` to SSOT canonical.
      - If strict for an entity type and no mapping exists, drop that entity.

    This is intended to ensure downstream consumers only see canonical SSOT codes.
    """

    def __init__(self, config: Dict[Text, Any]) -> None:
        self._config = config or {}

        ssot_dir = Path(str(self._config.get("ssot_dir", "src/shared/SSOT")))
        self._ssot_dir = ssot_dir

        # Entity -> SSOT file mapping
        mapping_any = self._config.get(
            "entity_ssot_files",
            {
                "metric": "MetricType.yml",
                "chart_type": "ChartType.yml",
                "group_by": "GroupByType.yml",
                "operator_type": "OperatorType.yml",
                "sex": "SexType.yml",
                "stroke_type": "StrokeType.yml",
                "boolean_type": "BooleanType.yml",
                "statistical_test_type": "StatisticalTestType.yml",
            },
        )
        self._entity_ssot_files: Dict[str, str] = {
            str(k): str(v) for k, v in cast(Dict[str, Any], mapping_any).items()
        }

        # Which entities are strict (unmapped values are dropped). Default: only `metric`.
        strict_any = self._config.get("strict_entities", ["metric"])
        self._strict_entities: Set[str] = set(ssot_yaml.as_str_list(strict_any))

        debug_any = self._config.get("debug", False)
        self._debug = bool(debug_any)

        self._indexes: Dict[str, _SSOTIndex] = {}
        self._load_indexes()

    def _load_indexes(self) -> None:
        for entity_name, fname in self._entity_ssot_files.items():
            fpath = self._ssot_dir / fname
            if not fpath.exists():
                if self._debug:
                    logger.warning(f"SSOT file missing for {entity_name}: {fpath}")
                continue
            try:
                self._indexes[entity_name] = _load_ssot_index(fpath)
                if self._debug:
                    logger.info(
                        f"Loaded SSOT index for {entity_name} from {fpath} ({len(self._indexes[entity_name].canonicals)} canonicals)"
                    )
            except Exception as e:
                logger.warning(f"Failed loading SSOT file {fpath}: {e}")

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: Any,
        resource: Any,
        execution_context: Any,
    ) -> "SSOTCanonicalizer":
        return cls(config)

    def _drop_entities_subsumed_by_metric(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # A lookup-table match that falls entirely inside a metric entity's own
        # span is just a substring of that metric's SSOT synonym text (e.g. "day"
        # inside HOSPITALIZED_IN's "first-day bed type") -- not independent
        # evidence of a separate entity, and misleading downstream (the LLM
        # planner sometimes honors it as a real request).
        metric_spans = [
            (cast(int, e["start"]), cast(int, e["end"]))
            for e in entities
            if e.get("entity") == "metric"
            and isinstance(e.get("start"), int)
            and isinstance(e.get("end"), int)
        ]
        if not metric_spans:
            return entities

        filtered: List[Dict[str, Any]] = []
        for ent in entities:
            if ent.get("entity") != "metric" and isinstance(ent.get("start"), int) and isinstance(ent.get("end"), int):
                start, end = cast(int, ent["start"]), cast(int, ent["end"])
                if any(m_start <= start and end <= m_end for m_start, m_end in metric_spans):
                    if self._debug:
                        logger.info(
                            f"Dropping {ent.get('entity')} entity subsumed by metric span: {ent.get('value')!r}"
                        )
                    continue
            filtered.append(ent)
        return filtered

    def process(self, messages: List[Message]) -> List[Message]:  # type: ignore[override]
        for message_any in cast(List[Any], messages):
            entities_any = message_any.get("entities")
            if not isinstance(entities_any, list) or not entities_any:
                continue

            new_entities: List[Dict[str, Any]] = []
            for ent_any in cast(List[Any], entities_any):
                if not isinstance(ent_any, dict):
                    continue
                ent = dict(cast(Dict[str, Any], ent_any))

                entity_name = str(ent.get("entity") or "")
                if not entity_name:
                    new_entities.append(ent)
                    continue

                idx = self._indexes.get(entity_name)
                if idx is None:
                    new_entities.append(ent)
                    continue

                raw_val = ent.get("value")
                if not isinstance(raw_val, str):
                    new_entities.append(ent)
                    continue

                mapped = idx.lookup(raw_val)
                if mapped is None:
                    if entity_name in self._strict_entities:
                        if self._debug:
                            logger.info(
                                f"Dropping non-SSOT {entity_name} value: {raw_val!r}"
                            )
                        continue
                    new_entities.append(ent)
                    continue

                if mapped != raw_val:
                    ent["_ssot_raw_value"] = raw_val
                    ent["value"] = mapped
                new_entities.append(ent)

            new_entities = self._drop_entities_subsumed_by_metric(new_entities)
            message_any.set("entities", new_entities)

        return messages
