# pyright: reportMissingTypeStubs=false, reportMissingModuleSource=false, reportUntypedClassDecorator=false, reportUntypedBaseClass=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Text, cast

import yaml  # type: ignore[import-untyped]  # pyright: ignore[reportMissingModuleSource, reportMissingTypeStubs]
from rasa.engine.graph import GraphComponent  # type: ignore
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # type: ignore
from rasa.shared.nlu.training_data.message import Message  # type: ignore

logger = logging.getLogger(__name__)


def _norm_text(text: str) -> str:
    # Conservative normalization: keep non-latin characters, but normalize spacing and common separators.
    s = text.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for v in cast(List[Any], value):
            if v is None:
                continue
            out.append(str(v))
        return out
    return [str(value)]


def _collect_synonyms(value: Any) -> List[str]:
    if isinstance(value, dict):
        out: List[str] = []
        for localized in cast(Dict[Any, Any], value).values():
            out.extend(_as_list(localized))
        return out
    return _as_list(value)


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

    Expected SSOT shape: a YAML list of items with keys like:
      - canonical: <CODE>
        synonyms: ["...", "..."]

    For MetricType.yml, some items also include data_type: Enum with an Enum list; we include
    enum keys + their synonyms as valid synonyms for that canonical as well.
    """

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        raw_list = cast(List[Any], raw)
        items = [
            cast(Dict[str, Any], item) for item in raw_list if isinstance(item, dict)
        ]
    elif isinstance(raw, dict):
        # Be permissive; allow a dict form if present.
        items = [cast(Dict[str, Any], raw)]

    canonicals: Set[str] = set()
    by_synonym: Dict[str, str] = {}

    for item in items:
        canonical_any = item.get("canonical")
        if not canonical_any:
            continue
        canonical = str(canonical_any)
        canonicals.add(canonical)

        synonyms = _collect_synonyms(item.get("synonyms"))
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
                        _collect_synonyms(cast(Dict[str, Any], e).get("synonyms"))
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
      - Also supports migrating old entity name `kpi` -> `metric`.

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
        self._strict_entities: Set[str] = set(_as_list(strict_any))

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

                # Migration shim: kpi -> metric
                if entity_name == "kpi":
                    ent["entity"] = "metric"
                    entity_name = "metric"

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

            message_any.set("entities", new_entities)

        return messages
