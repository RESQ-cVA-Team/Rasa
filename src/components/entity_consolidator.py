# pyright: reportUntypedClassDecorator=false, reportUntypedBaseClass=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import logging
from typing import Any, Dict, List, Optional, Text, Tuple, cast

from rasa.engine.graph import ExecutionContext, GraphComponent  # type: ignore
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # type: ignore
from rasa.engine.storage.resource import Resource  # type: ignore
from rasa.engine.storage.storage import ModelStorage  # type: ignore
from rasa.shared.nlu.training_data.message import Message  # type: ignore

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False)
class EntityConsolidator(GraphComponent):
    """Consolidates duplicate entities extracted from the same message."""

    def __init__(self, config: Dict[Text, Any]) -> None:
        self._config = config
        self._position_matching = config.get("position_matching", "exact")
        self._overlap_threshold = config.get("overlap_threshold", 0.5)
        self._position_tolerance = config.get("position_tolerance", 0)
        self._consolidation_key = config.get("consolidation_key", ["entity", "start", "end", "role", "value"])
        self._preserve_all_extractors = config.get("preserve_all_extractors", True)
        self._confidence_strategy = config.get("confidence_strategy", "highest")
        self._role_aware = config.get("role_aware", True)
        self._value_normalization = config.get("value_normalization", False)
        self._debug_logging = config.get("debug_logging", False)
        self._collect_stats = config.get("collect_stats", False)
        self._stats: Dict[str, float] = {"total_processed": 0.0, "total_consolidated": 0.0, "consolidation_ratio": 0.0}

        if self._position_matching not in ["exact", "overlap", "ignore"]:
            raise ValueError(f"Invalid position_matching: {self._position_matching}")

        if not 0.0 <= self._overlap_threshold <= 1.0:
            raise ValueError(f"overlap_threshold must be between 0.0 and 1.0, got {self._overlap_threshold}")

        if self._position_tolerance < 0:
            raise ValueError(f"position_tolerance must be >= 0, got {self._position_tolerance}")

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "EntityConsolidator":
        return cls(config)

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            entities = message.get("entities", [])
            if entities:
                if self._debug_logging:
                    logger.info(f"Before consolidation: {len(entities)} entities")
                    for i, ent in enumerate(entities):
                        logger.info(f"  {i}: {ent.get('entity')}={ent.get('value')} [{ent.get('start')}-{ent.get('end')}] role={ent.get('role')}")

                consolidated = self._consolidate_entities(entities)
                message.set("entities", consolidated)

                if self._debug_logging:
                    logger.info(f"After consolidation: {len(consolidated)} entities")
                    for i, ent in enumerate(consolidated):
                        logger.info(f"  {i}: {ent.get('entity')}={ent.get('value')} [{ent.get('start')}-{ent.get('end')}] role={ent.get('role')} extractors={len(ent.get('extractors', []))}")
        return messages

    def _normalize_value(self, value: Any) -> Any:
        """Normalize entity value if configured to do so."""
        if not self._value_normalization or not isinstance(value, str):
            return value
        return value.lower().strip()

    def _positions_match(self, ent1: Dict[str, Any], ent2: Dict[str, Any]) -> bool:
        """Check if two entities match based on position matching strategy."""
        if self._position_matching == "ignore":
            return True

        start1, end1 = ent1.get("start"), ent1.get("end")
        start2, end2 = ent2.get("start"), ent2.get("end")

        if start1 is None or end1 is None or start2 is None or end2 is None:
            return True

        if self._position_matching == "exact":
            tolerance = self._position_tolerance
            return abs(start1 - start2) <= tolerance and abs(end1 - end2) <= tolerance

        elif self._position_matching == "overlap":
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)

            if overlap_start >= overlap_end:
                return False

            overlap_length = overlap_end - overlap_start
            min_length = min(end1 - start1, end2 - start2)

            if min_length == 0:
                return overlap_length > 0

            overlap_ratio = overlap_length / min_length
            return overlap_ratio >= self._overlap_threshold

        return False

    def _generate_key(self, ent: Dict[str, Any]) -> Tuple[Any, ...]:
        """Generate consolidation key based on configuration."""
        key_parts: List[Any] = []

        for key_component in self._consolidation_key:
            if key_component == "entity":
                key_parts.append(ent.get("entity"))
            elif key_component == "value":
                value = ent.get("value")
                key_parts.append(self._normalize_value(value))
            elif key_component == "role":
                if self._role_aware:
                    key_parts.append(ent.get("role"))
            elif key_component == "start":
                if self._position_matching == "exact":
                    key_parts.append(ent.get("start"))
            elif key_component == "end":
                if self._position_matching == "exact":
                    key_parts.append(ent.get("end"))
            elif key_component == "position_range":
                start, end = ent.get("start"), ent.get("end")
                if start is not None and end is not None:
                    range_key = f"{start // 10}-{end // 10}"
                    key_parts.append(range_key)

        return tuple(key_parts)

    def _consolidate_entities(self, entities: List[Dict[Text, Any]]) -> List[Dict[Text, Any]]:
        """Consolidate entities based on configuration."""
        original_count = len(entities)

        if self._position_matching in ["exact", "ignore"]:
            result = self._consolidate_by_key(entities)
        else:
            result = self._consolidate_by_overlap(entities)

        if self._collect_stats:
            self._stats["total_processed"] = float(self._stats.get("total_processed", 0.0)) + float(original_count)
            self._stats["total_consolidated"] = float(self._stats.get("total_consolidated", 0.0)) + float(original_count - len(result))
            if self._stats["total_processed"] > 0:
                self._stats["consolidation_ratio"] = self._stats["total_consolidated"] / self._stats["total_processed"]

            if self._debug_logging:
                logger.info(f"Consolidation stats: {self._stats}")

        return result

    def _consolidate_by_key(self, entities: List[Dict[Text, Any]]) -> List[Dict[Text, Any]]:
        """Consolidate entities using key-based matching."""
        consolidated: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

        for ent in entities:
            key = self._generate_key(ent)

            if key not in consolidated:
                consolidated[key] = self._create_consolidated_entity(ent)
            else:
                self._merge_entity_data(consolidated[key], ent)

        return list(consolidated.values())

    def _consolidate_by_overlap(self, entities: List[Dict[Text, Any]]) -> List[Dict[Text, Any]]:
        """Consolidate entities using overlap-based matching."""
        consolidated: List[Dict[str, Any]] = []

        for ent in entities:
            merged = False

            for existing in consolidated:
                if ent.get("entity") == existing.get("entity") and (not self._role_aware or ent.get("role") == existing.get("role")) and self._normalize_value(ent.get("value")) == self._normalize_value(existing.get("value")) and self._positions_match(ent, existing):
                    self._merge_entity_data(existing, ent)
                    merged = True
                    break

            if not merged:
                consolidated.append(self._create_consolidated_entity(ent))

        return consolidated

    def _create_consolidated_entity(self, ent: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new consolidated entity from the first occurrence."""
        out: Dict[str, Any] = {
            "entity": ent.get("entity"),
            "start": ent.get("start"),
            "end": ent.get("end"),
            "value": ent.get("value"),
            # Single-string key Rasa's own built-in evaluator expects
            # (entities_by_extractors[p["extractor"]] in align_entity_predictions) --
            # first-write-wins, matching how start/end/value are already handled here.
            # Additive only: the richer per-extractor "extractors" list below is unchanged.
            "extractor": ent.get("extractor"),
            "extractors": [],
            "role_extractors": [],
        }

        if self._role_aware and ent.get("role"):
            out["role"] = ent.get("role")

        self._add_extractor_info(out, ent)
        return out

    def _merge_entity_data(self, consolidated: Dict[str, Any], ent: Dict[str, Any]) -> None:
        """Merge data from a new entity into an existing consolidated entity."""
        self._add_extractor_info(consolidated, ent)

        # Update position if needed (e.g., take the span that covers both)
        if self._position_matching == "overlap":
            start = min(consolidated.get("start", float("inf")), ent.get("start", float("inf")))
            end = max(consolidated.get("end", 0), ent.get("end", 0))
            if start != float("inf"):
                consolidated["start"] = start
            if end != 0:
                consolidated["end"] = end

    def _add_extractor_info(self, consolidated: Dict[str, Any], ent: Dict[str, Any]) -> None:
        """Add extractor information to consolidated entity."""
        if not self._preserve_all_extractors:
            return

        # Add entity extractor info
        extractor = ent.get("extractor")
        confidence: Optional[float] = None
        c_val = ent.get("confidence_entity") or ent.get("confidence")
        if isinstance(c_val, (int, float)):
            confidence = float(c_val)
        if extractor:
            extractor_info: Dict[str, Any] = {"extractor": extractor, "confidence": confidence}
            extractors_list = cast(List[Dict[str, Any]], consolidated.get("extractors") or [])
            if extractor_info not in extractors_list:
                extractors_list.append(extractor_info)
                consolidated["extractors"] = extractors_list

        # Add role extractor info
        role_extractor = ent.get("role_extractor")
        role_confidence: Optional[float] = None
        rc_val = ent.get("confidence_role")
        if isinstance(rc_val, (int, float)):
            role_confidence = float(rc_val)
        if role_extractor:
            role_info: Dict[str, Any] = {"extractor": role_extractor, "confidence": role_confidence}
            role_extractors_list = cast(List[Dict[str, Any]], consolidated.get("role_extractors") or [])
            if role_info not in role_extractors_list:
                role_extractors_list.append(role_info)
                consolidated["role_extractors"] = role_extractors_list

        self._recompute_confidences(consolidated)

    def _aggregate_confidence(self, values: List[Optional[float]]) -> Tuple[Optional[float], Optional[List[float]]]:
        clean_vals: List[float] = [float(v) for v in values if isinstance(v, (int, float))]
        if not clean_vals:
            return None, None
        if self._confidence_strategy == "average":
            return sum(clean_vals) / len(clean_vals), None
        if self._confidence_strategy == "all":
            return max(clean_vals), clean_vals
        return max(clean_vals), None

    def _recompute_confidences(self, consolidated: Dict[str, Any]) -> None:
        # Entity confidence aggregation
        ent_conf_values: List[Optional[float]] = []
        extractors_list = cast(List[Dict[str, Any]], consolidated.get("extractors") or [])
        for ent_info in extractors_list:
            ent_conf: Optional[float] = None
            c = ent_info.get("confidence")
            if isinstance(c, (int, float)):
                ent_conf = float(c)
            ent_conf_values.append(ent_conf)
        agg_ent, all_ent = self._aggregate_confidence(ent_conf_values)
        if agg_ent is not None:
            consolidated["confidence_entity"] = agg_ent
        else:
            consolidated.pop("confidence_entity", None)
        if self._confidence_strategy == "all":
            if all_ent is not None:
                consolidated["confidence_entity_all"] = all_ent
            else:
                consolidated.pop("confidence_entity_all", None)
        else:
            consolidated.pop("confidence_entity_all", None)

        # Role confidence aggregation (only if role is present)
        if self._role_aware and consolidated.get("role") is not None:
            role_conf_values: List[Optional[float]] = []
            role_extractors_list = cast(List[Dict[str, Any]], consolidated.get("role_extractors") or [])
            for role_info in role_extractors_list:
                role_conf: Optional[float] = None
                c = role_info.get("confidence")
                if isinstance(c, (int, float)):
                    role_conf = float(c)
                role_conf_values.append(role_conf)
            agg_role, all_role = self._aggregate_confidence(role_conf_values)
            if agg_role is not None:
                consolidated["confidence_role"] = agg_role
            else:
                consolidated.pop("confidence_role", None)
            if self._confidence_strategy == "all":
                if all_role is not None:
                    consolidated["confidence_role_all"] = all_role
                else:
                    consolidated.pop("confidence_role_all", None)
            else:
                consolidated.pop("confidence_role_all", None)
        else:
            consolidated.pop("confidence_role", None)
            consolidated.pop("confidence_role_all", None)
