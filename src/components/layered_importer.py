# pyright: reportMissingTypeStubs=false, reportMissingModuleSource=false
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast

import yaml  # type: ignore[import-untyped]  # pyright: ignore[reportMissingModuleSource, reportMissingTypeStubs]
from rasa.shared.core.domain import Domain  # type: ignore
from rasa.shared.core.training_data.structures import StoryGraph  # type: ignore
from rasa.shared.importers.importer import TrainingDataImporter  # type: ignore
from rasa.shared.nlu.training_data import loading as nlu_loading  # type: ignore

from src.components import ssot_yaml

try:  # story reader imports (robust across minor Rasa versions)
    from rasa.shared.core.training_data.story_reader.yaml_story_reader import YAMLStoryReader  # type: ignore
except Exception:  # pragma: no cover
    YAMLStoryReader = None  # type: ignore

ADD = "add"
REPLACE = "replace"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OverlayImporter")


def _iter_yaml_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in {".yml", ".yaml"} else []
    if path.is_dir():
        return [p for p in path.rglob("*") if p.suffix.lower() in {".yml", ".yaml"}]
    return []


def _load_yaml_docs(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    files: List[Path] = []
    for p in paths:
        if p.exists():
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                files.extend(_iter_yaml_files(p))
    for fpath in files:
        with fpath.open("r", encoding="utf-8") as f:
            doc_any = yaml.safe_load(f)
            if isinstance(doc_any, dict):
                docs.append(cast(Dict[str, Any], doc_any))
    return docs


def _has_yaml_under(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() in {".yml", ".yaml"}
    if path.is_dir():
        for p in path.rglob("*"):
            if p.suffix.lower() in {".yml", ".yaml"}:
                return True
    return False


def _parse_key(key: str, inherited_op: str) -> Tuple[str, str]:
    if key.endswith(".add"):
        return key[:-4], ADD
    if key.endswith(".replace"):
        return key[:-8], REPLACE
    return key, inherited_op


def _normalize_ops(node: Any, inherited: str = REPLACE) -> Tuple[Any, str]:
    if isinstance(node, dict):
        clean: Dict[str, Any] = {}
        dict_node = cast(Dict[str, Any], node)
        for k, v in dict_node.items():
            base_k, child_op = _parse_key(k, inherited)
            sub_clean, _ = _normalize_ops(v, child_op)
            clean[base_k] = sub_clean
        return clean, inherited
    if isinstance(node, list):
        list_node = cast(List[Any], node)
        return [_normalize_ops(i, inherited)[0] for i in list_node], inherited
    return node, inherited


def _list_unique_extend(base: List[Any], extra: List[Any]) -> List[Any]:
    seen: Set[str] = set()
    out: List[Any] = []
    for x in base + extra:
        key = yaml.dump(x, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def _deep_add(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        base_dict = cast(Dict[str, Any], base)
        overlay_dict = cast(Dict[str, Any], overlay)
        out: Dict[str, Any] = dict(base_dict)
        for k, v in overlay_dict.items():
            if k in out:
                out[k] = _deep_add(out[k], v)
            else:
                out[k] = v
        return out
    if isinstance(base, list) and isinstance(overlay, list):
        return _list_unique_extend(cast(List[Any], base), cast(List[Any], overlay))
    return overlay


def _apply_overlay_strict_dict(base: Optional[Dict[str, Any]], overlay: Dict[str, Any], op: str, section_name: str = "") -> Dict[str, Any]:
    if base is None:
        base = {}
    if op == REPLACE:
        missing = [k for k in overlay.keys() if k not in base]
        if missing:
            sec = f" in '{section_name}'" if section_name else ""
            raise ValueError(f"Overlay attempted to replace non-existent keys{sec}: {missing}")
        merged = dict(base)
        merged.update(overlay)
        return merged
    return _deep_add(base, overlay)


DOMAIN_LIST_KEYS = {"intents", "entities", "actions", "e2e_actions"}


def _merge_domain_docs(base_docs: List[Dict[str, Any]], overlay_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    base: Dict[str, Any] = {}
    for d in base_docs:
        clean, _ = _normalize_ops(d, REPLACE)
        base = _deep_add(base, clean)

    for d in overlay_docs:
        # Normalize to strip markers, but capture per-section ops
        clean_top: Dict[str, Any] = {}
        section_ops: Dict[str, str] = {}
        for raw_key, value in d.items():
            key, sec_op = _parse_key(raw_key, REPLACE)
            section_ops[key] = sec_op
            clean_top[key] = _normalize_ops(value, sec_op)[0]

        # Dict-like sections
        for section in ("responses", "slots", "forms", "session_config"):
            if section in clean_top:
                base_section = base.get(section, {})
                op = section_ops.get(section, REPLACE)
                base[section] = _apply_overlay_strict_dict(base_section, clean_top[section], op, section)

        # List-like sections
        for section in DOMAIN_LIST_KEYS:
            if section in clean_top:
                op = section_ops.get(section, REPLACE)
                if section not in base:
                    if op == REPLACE:
                        raise ValueError(f"Overlay attempted to replace non-existent domain section '{section}'")
                    base[section] = []
                if op == REPLACE:
                    base[section] = clean_top[section]
                else:
                    base[section] = _list_unique_extend(base.get(section, []), clean_top[section])

        # Any other keys -> deep add
        for k, v in clean_top.items():
            if k in {"responses", "slots", "forms", "session_config"} | DOMAIN_LIST_KEYS:
                continue
            base[k] = _deep_add(base.get(k), v)

    return base


# NLU list items are keyed by one of these type markers (optionally suffixed with
# ".add"/".replace", e.g. "lookup.add"). Anything outside this set (intent-only,
# historically) was silently dropped by the merge below, which meant lookup/synonym/
# regex tables never survived the base+overlay merge despite being valid Rasa NLU YAML.
_NLU_ITEM_TYPES = ("intent", "lookup", "synonym", "regex")


def _split_nlu_item_op(
    item: Dict[str, Any], inherited: str = REPLACE
) -> Tuple[Dict[str, Any], str, Optional[str], Optional[str]]:
    type_key = next(
        (k for k in item.keys() if k in _NLU_ITEM_TYPES or any(k.startswith(f"{t}.") for t in _NLU_ITEM_TYPES)),
        None,
    )
    if not type_key:
        return item, inherited, None, None
    base_key, op = _parse_key(type_key, inherited)
    new_item = dict(item)
    new_item[base_key] = new_item.pop(type_key)
    return new_item, op, base_key, cast(Optional[str], new_item.get(base_key))


def _merge_nlu_docs(base_docs: List[Dict[str, Any]], overlay_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    version = "3.1"

    def _feed(doc: Dict[str, Any]):
        nonlocal version
        if "version" in doc:
            version = doc["version"]
        for it in cast(List[Any], doc.get("nlu") or []):
            if isinstance(it, dict):
                item = cast(Dict[str, Any], it)
                item_type = next((t for t in _NLU_ITEM_TYPES if t in item), None)
                if item_type:
                    by_key.setdefault((item_type, cast(str, item[item_type])), []).append(item)

    for d in base_docs:
        clean, _ = _normalize_ops(d, REPLACE)
        _feed(clean)

    for d in overlay_docs:
        clean, parent_op = _normalize_ops(d, REPLACE)
        for raw in cast(List[Any], clean.get("nlu") or []):
            item_raw = cast(Dict[str, Any], raw)
            item, item_op, item_type, name = _split_nlu_item_op(item_raw, inherited=parent_op)
            if not item_type or not name:
                continue
            key = (item_type, name)
            if item_op == REPLACE:
                if key not in by_key:
                    raise ValueError(f"Overlay attempted to replace unknown {item_type} '{name}'")
                by_key[key] = [item]
            else:
                by_key.setdefault(key, []).append(item)

    merged_items: List[Dict[str, Any]] = []
    for items in by_key.values():
        merged_items.extend(items)
    return {"version": version, "nlu": merged_items}


# Entity -> SSOT file mapping for auto-generated NLU lookup/synonym data.
# Mirrors ssot_canonicalizer.py's entity_ssot_files (which canonicalizes
# *after* extraction); this generates the lookup/synonym tables that help
# extraction find the spans in the first place, so RegexEntityExtractor and
# EntitySynonymMapper never need hand-typed, per-locale duplicates of SSOT
# data. Add a new SSOT type here and every locale picks it up on the next
# train -- no Rasa-side lookup file to write or keep in sync.
DEFAULT_SSOT_NLU_ENTITIES: Dict[str, str] = {
    "stroke_type": "StrokeType.yml",
    "group_by": "GroupByType.yml",
    "metric": "MetricType.yml",
    "country_code": "CountryType.yml",
    "statistical_test_type": "StatisticalTestType.yml",
    "chart_type": "ChartType.yml",
}


def _detect_locale(domain_paths: List[Path]) -> str:
    """Best-effort locale code from an overlay domain path like src/locales/cs/CZ/domain."""
    for p in domain_paths:
        parts = p.parts
        if "locales" in parts:
            idx = parts.index("locales")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return "en"


def _default_ssot_dir(domain_paths: List[Path]) -> Path:
    for p in domain_paths:
        parts = p.parts
        if "src" in parts:
            idx = parts.index("src")
            root = Path(*parts[:idx]) if idx > 0 else Path(".")
            candidate = root / "src" / "shared" / "SSOT"
            if candidate.exists():
                return candidate
    return Path("src/shared/SSOT")


def _build_ssot_nlu_doc(ssot_dir: Path, locale: str, entities: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Synthesize `lookup`/`synonym` NLU items straight from SSOT for one locale."""
    nlu_items: List[Dict[str, Any]] = []
    for entity_name, filename in entities.items():
        fpath = ssot_dir / filename
        if not fpath.exists():
            continue
        try:
            ssot_items = ssot_yaml.load_ssot_items(fpath)
        except Exception as e:
            logger.warning(f"Failed loading SSOT file {fpath} for NLU generation: {e}")
            continue

        lookup_examples: List[str] = []
        synonym_items: List[Dict[str, Any]] = []
        for entry in ssot_items:
            canonical_any = entry.get("canonical")
            if not isinstance(canonical_any, str) or not canonical_any.strip():
                continue
            canonical = canonical_any.strip().upper()
            names = ssot_yaml.locale_synonyms(entry, locale)
            if not names:
                continue
            # country_code canonicals are bare ISO 3166-1 alpha-2 codes (TO, US,
            # IN, IT, BY, ...). RegexEntityExtractor's lookup-table matching is
            # case-insensitive (case_sensitive: false in config.yml, needed so
            # the *name* synonyms below still match lowercase user text), which
            # made 14 of these codes collide with ordinary English function
            # words ("to", "us", "in", "it", "is", "at", "as", "by", "be", "so",
            # "am", "me", "do", "my") and 8 more with Czech ones ("to", "za",
            # "do", "ve", "se", "si", "my", "mu") -- e.g. "...from 2023-01-01 to
            # 2023-12-31" got a spurious country_code=TO (Tonga) entity from the
            # word "to", which then broke hospital-scope resolution downstream.
            # Country *names* have no such collision risk, so only the bare
            # code is excluded here; DIET still recognizes codes contextually
            # from the real "[US](country_code)"-style training examples.
            if entity_name != "country_code":
                lookup_examples.append(canonical)
            lookup_examples.extend(names)
            synonym_items.append(
                {"synonym": canonical, "examples": "".join(f"- {n}\n" for n in names)}
            )

        if lookup_examples:
            nlu_items.append(
                {"lookup": entity_name, "examples": "".join(f"- {e}\n" for e in lookup_examples)}
            )
        nlu_items.extend(synonym_items)

    if not nlu_items:
        return None
    return {"version": "3.1", "nlu": nlu_items}


def _merge_config_docs(base_docs: List[Dict[str, Any]], overlay_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge Rasa config dictionaries with add/replace semantics.

    Semantics:
      * Keys ending with .add / .replace control list/dict merging like domain logic.
      * Known list sections: pipeline, policies. `.replace` fully replaces, `.add` appends with de-dup.
      * Dict (top-level) keys: deep add by default; `.replace` updates only existing keys (strict like domain replace).
    """
    # Start from combined base (deep add)
    merged: Dict[str, Any] = {}
    for d in base_docs:
        clean, _ = _normalize_ops(d, REPLACE)
        merged = _deep_add(merged, clean)

    def _merge_in(doc: Dict[str, Any]) -> None:
        clean_top: Dict[str, Any] = {}
        section_ops: Dict[str, str] = {}
        for raw_key, value in doc.items():
            key, sec_op = _parse_key(raw_key, REPLACE)
            section_ops[key] = sec_op
            clean_top[key] = _normalize_ops(value, sec_op)[0]

        # List-like sections
        for section in ("pipeline", "policies"):
            if section in clean_top:
                op = section_ops.get(section, REPLACE)
                if op == REPLACE:
                    merged[section] = clean_top[section]
                else:
                    existing_raw = merged.get(section, [])
                    existing_list = cast(List[Any], existing_raw) if isinstance(existing_raw, list) else []
                    new_raw = clean_top[section]
                    new_list = cast(List[Any], new_raw) if isinstance(new_raw, list) else [new_raw]
                    merged[section] = _list_unique_extend(existing_list, new_list)

        # Other keys
        for k, v in clean_top.items():
            if k in {"pipeline", "policies"}:
                continue
            op = section_ops.get(k, REPLACE)
            if isinstance(v, dict):
                base_section_any = merged.get(k, {})
                base_section = cast(Dict[str, Any], base_section_any) if isinstance(base_section_any, dict) else {}
                if op == REPLACE:
                    # Only replace existing keys strictly
                    missing: List[str] = []
                    for mk in list(cast(Dict[str, Any], v).keys()):
                        if mk not in base_section:
                            missing.append(mk)
                    if base_section and missing:
                        raise ValueError(f"Config overlay attempted to replace non-existent keys in '{k}': {missing}")
                    merged[k] = {**base_section, **v}
                else:
                    merged[k] = _deep_add(base_section, v)
            else:
                # Scalar or list (non-recognized list handled by deep add semantics)
                if op == REPLACE:
                    # Scalars always take the overlay value outright, even for keys
                    # not yet present in the base -- REPLACE's strict-keys check
                    # only applies to dict/list sections above.
                    merged[k] = v
                else:
                    existing_any = merged.get(k)
                    if isinstance(existing_any, list) and isinstance(v, list):
                        merged[k] = _list_unique_extend(cast(List[Any], existing_any), cast(List[Any], v))
                    else:
                        merged[k] = v

    for d in overlay_docs:
        _merge_in(d)

    return merged


def _derive_nlu_paths(domain_paths: List[Path]) -> List[Path]:
    n_paths: List[Path] = []
    for d in domain_paths:
        root = d.parent
        for rel in (Path("data") / "nlu", Path("nlu")):
            cand = root / rel
            if cand.exists():
                n_paths.append(cand)
    return n_paths


def _parse_comma_paths(value: Optional[str]) -> List[Path]:
    if not value:
        return []
    out: List[Path] = []
    for s in [s.strip() for s in value.split(",") if s.strip()]:
        p = Path(s)
        if p.exists():
            out.append(p)
    return out


def _uniq_paths(paths: List[Path]) -> List[Path]:
    seen: Set[str] = set()
    out: List[Path] = []
    for x in paths:
        sx = str(x)
        if sx not in seen:
            seen.add(sx)
            out.append(x)
    return out


def _to_existing_strs(paths: List[Path]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if p.exists():
            if p.is_dir():
                out.append(str(p))
            elif p.is_file() and p.suffix.lower() in {".yml", ".yaml"}:
                out.append(str(p))
    return out


def _load_domain_as_dict(path: Path) -> Dict[str, Any]:
    loaded_domain = cast(Any, Domain).load(str(path))
    return cast(Dict[str, Any], loaded_domain.as_dict())


def _build_domain(domain_data: Dict[str, Any]) -> Any:
    return cast(Any, Domain).from_dict(domain_data)


def _load_training_data(path: Path) -> Any:
    return cast(Any, nlu_loading).load_data(str(path))


def _build_story_graph(steps: List[Any]) -> Any:
    return cast(Any, StoryGraph)(steps)


class OverlayImporter(TrainingDataImporter):  # pyright: ignore[reportUntypedBaseClass]
    def __init__(self, *args: Any, base_domain: Optional[List[str]] = None, overlay_domain: Optional[List[str]] = None, **kwargs: Any):
        cfg: Dict[str, Any] = {}
        if args and isinstance(args[0], dict):
            cfg.update(cast(Dict[str, Any], args[0]))
        cfg.update(kwargs)

        base_domain = base_domain if base_domain is not None else cfg.get("base_domain")
        overlay_domain = overlay_domain if overlay_domain is not None else cfg.get("overlay_domain")

        self._base_domain_paths = [Path(p) for p in (base_domain or [])]
        self._overlay_domain_paths = [Path(p) for p in (overlay_domain or [])]
        logger.info(f"Base domain files: {[str(p) for p in self._base_domain_paths]}")
        logger.info(f"Overlay domain files: {[str(p) for p in self._overlay_domain_paths]}")

        # Allow env to override base/overlay domains dynamically
        env_base_domain = os.environ.get("OVERLAY_BASE_DOMAIN", "").strip()
        if env_base_domain:
            override_paths = _parse_comma_paths(env_base_domain)
            if override_paths:
                self._base_domain_paths = override_paths

        # Allow env to override overlay domains dynamically for CI/builds
        env_overlay_domain = os.environ.get("OVERLAY_DOMAIN", "").strip()
        if env_overlay_domain:
            override_paths = _parse_comma_paths(env_overlay_domain)
            if override_paths:
                self._overlay_domain_paths = override_paths

        self._base_nlu_paths: List[Path] = _derive_nlu_paths(self._base_domain_paths)
        self._overlay_nlu_paths: List[Path] = _derive_nlu_paths(self._overlay_domain_paths)

        raw_overlay_nlu: Any = cfg.get("overlay_nlu")
        env_overlay_nlu: List[str] = []
        if isinstance(raw_overlay_nlu, str):
            env_overlay_nlu = [p.strip() for p in raw_overlay_nlu.split(",") if p.strip()]
        elif isinstance(raw_overlay_nlu, list):
            for p in cast(List[Any], raw_overlay_nlu):
                s = str(p).strip()
                if s:
                    env_overlay_nlu.append(s)
        for p in env_overlay_nlu:
            pp = Path(p)
            if pp.exists():
                self._overlay_nlu_paths.append(pp)

        env_str = os.environ.get("OVERLAY_NLU", "").strip()
        if env_str:
            for pp in _parse_comma_paths(env_str):
                self._overlay_nlu_paths.append(pp)

        self._base_nlu_paths = _uniq_paths(self._base_nlu_paths)
        self._overlay_nlu_paths = _uniq_paths(self._overlay_nlu_paths)
        if self._base_nlu_paths:
            logger.info(f"Base NLU paths: {[str(p) for p in self._base_nlu_paths]}")
        if self._overlay_nlu_paths:
            logger.info(f"Overlay NLU paths: {[str(p) for p in self._overlay_nlu_paths]}")

        # Stories (rules + stories) layering: discover base/overlay story roots (data directories)
        def _story_roots_from_domain_paths(paths: List[Path]) -> List[Path]:
            outs: List[Path] = []
            for p in paths:
                root = p.parent / "data"
                if root.exists() and root.is_dir():
                    outs.append(root)
            return outs

        self._base_story_roots: List[Path] = _story_roots_from_domain_paths(self._base_domain_paths)
        self._overlay_story_roots: List[Path] = _story_roots_from_domain_paths(self._overlay_domain_paths)

        # Allow env to specify explicit overlay story roots (comma separated)
        env_story = os.environ.get("OVERLAY_STORIES", "").strip()
        if env_story:
            extra = _parse_comma_paths(env_story)
            for p in extra:
                if p not in self._overlay_story_roots:
                    self._overlay_story_roots.append(p)

        if self._base_story_roots:
            logger.info(f"Base story roots: {[str(p) for p in self._base_story_roots]}")
        if self._overlay_story_roots:
            logger.info(f"Overlay story roots: {[str(p) for p in self._overlay_story_roots]}")

        # Config layering: detect config.yml adjacent to domain roots
        def _config_from_domain_paths(paths: List[Path]) -> List[Path]:
            outs: List[Path] = []
            for p in paths:
                root = p.parent
                cand = root / "config.yml"
                if cand.exists():
                    outs.append(cand)
            return outs

        self._base_config_paths: List[Path] = _config_from_domain_paths(self._base_domain_paths)
        self._overlay_config_paths: List[Path] = _config_from_domain_paths(self._overlay_domain_paths)

        # Env overrides for config layering
        env_base_cfg = os.environ.get("OVERLAY_BASE_CONFIG", "").strip()
        if env_base_cfg:
            base_cfg_paths = _parse_comma_paths(env_base_cfg)
            if base_cfg_paths:
                self._base_config_paths = base_cfg_paths
        env_overlay_cfg = os.environ.get("OVERLAY_CONFIG", "").strip()
        if env_overlay_cfg:
            overlay_cfg_paths = _parse_comma_paths(env_overlay_cfg)
            if overlay_cfg_paths:
                self._overlay_config_paths = overlay_cfg_paths

        if self._base_config_paths:
            logger.info(f"Base config files: {[str(p) for p in self._base_config_paths]}")
        if self._overlay_config_paths:
            logger.info(f"Overlay config files: {[str(p) for p in self._overlay_config_paths]}")

        # SSOT-driven NLU lookup/synonym generation -- see DEFAULT_SSOT_NLU_ENTITIES.
        ssot_nlu_entities_any: Any = cfg.get("ssot_nlu_entities", DEFAULT_SSOT_NLU_ENTITIES)
        self._ssot_nlu_entities: Dict[str, str] = {
            str(k): str(v) for k, v in cast(Dict[str, Any], ssot_nlu_entities_any).items()
        }
        ssot_dir_cfg = cfg.get("ssot_dir")
        self._ssot_dir: Path = (
            Path(str(ssot_dir_cfg)) if ssot_dir_cfg else _default_ssot_dir(self._base_domain_paths)
        )
        self._ssot_locale: str = _detect_locale(self._overlay_domain_paths or self._base_domain_paths)
        logger.info(f"SSOT NLU generation: dir={self._ssot_dir} locale={self._ssot_locale} entities={list(self._ssot_nlu_entities)}")

    def get_domain(self) -> Any:
        base_docs: List[Dict[str, Any]] = []
        for p in self._base_domain_paths:
            if not _has_yaml_under(p):
                logger.info(f"Skipping base domain path with no YAML: {p}")
                continue
            logger.info(f"Loading base domain: {p}")
            base_docs.append(_load_domain_as_dict(p))
        overlay_docs: List[Dict[str, Any]] = []
        for p in self._overlay_domain_paths:
            if not _has_yaml_under(p):
                logger.info(f"Skipping overlay domain path with no YAML: {p}")
                continue
            logger.info(f"Loading overlay domain: {p}")
            overlay_docs.append(_load_domain_as_dict(p))
        logger.info("Merging domains...")
        merged = _merge_domain_docs(base_docs, overlay_docs)
        logger.info(f"Merged domain keys: {list(merged.keys())}")
        for ov_doc in overlay_docs:
            for k in ov_doc.keys():
                if k in merged:
                    logger.info(f"Overlay key '{k}' present in merged domain.")

        dump_target = os.environ.get("OVERLAY_DUMP_DOMAIN", "").strip()
        if dump_target:
            try:
                if dump_target.lower() in {"1", "true", "yes", "stdout"}:
                    yaml.safe_dump(merged, sys.stdout, sort_keys=False, allow_unicode=True)
                else:
                    out_path = Path(dump_target)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with out_path.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)
                    logger.info(f"Dumped merged domain to {out_path}")
            except Exception as e:
                logger.warning(f"Failed to dump merged domain: {e}")
        return _build_domain(merged)

    def get_nlu_data(self, language: Optional[str] = None) -> Any:
        base_paths = _to_existing_strs(self._base_nlu_paths)
        overlay_paths = _to_existing_strs(self._overlay_nlu_paths)

        if not base_paths and not overlay_paths:
            with tempfile.TemporaryDirectory(prefix="v2_empty_nlu_") as td:
                tmp = Path(td) / "empty_nlu.yml"
                with tmp.open("w", encoding="utf-8") as f:
                    yaml.safe_dump({"version": "3.1", "nlu": []}, f, sort_keys=False, allow_unicode=True)
                return _load_training_data(tmp)

        logger.info(f"Merging NLU from base={base_paths} overlays={overlay_paths}")
        base_docs = _load_yaml_docs([Path(p) for p in base_paths])
        overlay_docs = _load_yaml_docs([Path(p) for p in overlay_paths])

        ssot_doc = _build_ssot_nlu_doc(self._ssot_dir, self._ssot_locale, self._ssot_nlu_entities)
        if ssot_doc:
            base_docs = base_docs + [ssot_doc]
            logger.info(f"Injected SSOT-derived NLU data for locale={self._ssot_locale}")

        merged: Dict[str, Any] = _merge_nlu_docs(base_docs, overlay_docs)
        merged_nlu_list = cast(List[Dict[str, Any]], merged.get("nlu", []))
        intents = [str(it.get("intent")) for it in merged_nlu_list if it.get("intent")]
        logger.info(f"Merged NLU intents: {sorted(set(intents))}")

        dump_target = os.environ.get("OVERLAY_DUMP_NLU", "").strip()
        with tempfile.TemporaryDirectory(prefix="v2_merged_nlu_") as td:
            tmp = Path(td) / "merged_nlu.yml"
            with tmp.open("w", encoding="utf-8") as f:
                yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)

            if dump_target:
                try:
                    if dump_target.lower() in {"1", "true", "yes", "stdout"}:
                        yaml.safe_dump(merged, sys.stdout, sort_keys=False, allow_unicode=True)
                    else:
                        out_path = Path(dump_target)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        with out_path.open("w", encoding="utf-8") as outf:
                            yaml.safe_dump(merged, outf, sort_keys=False, allow_unicode=True)
                        logger.info(f"Dumped merged NLU to {out_path}")
                except Exception as e:
                    logger.warning(f"Failed to dump merged NLU: {e}")

            return _load_training_data(tmp)

    def get_stories(self, exclusion_percentage: Optional[int] = None) -> Any:
        # Aggregate story & rule files from base then overlays
        story_files: List[Path] = []

        def _collect(root: Path) -> None:
            if not root.exists():
                return
            candidates: List[Path] = []
            for sub in (root, root / "stories", root / "rules"):
                if sub.exists() and sub.is_dir():
                    for pattern in ("*.yml", "*.yaml"):
                        candidates.extend(list(sub.rglob(pattern)))
            for c in candidates:
                if c not in story_files:
                    story_files.append(c)

        for r in self._base_story_roots:
            _collect(r)
        for r in self._overlay_story_roots:
            _collect(r)

        if not story_files or YAMLStoryReader is None:
            return _build_story_graph([])

        all_steps: List[Any] = []
        for fp in story_files:
            try:
                reader = cast(Any, YAMLStoryReader)(domain=None)
                reader.read_from_file(str(fp))
                steps = getattr(reader, "story_steps", [])
                if steps and isinstance(steps, list):
                    all_steps.extend(cast(List[Any], steps))
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to load story file {fp}: {e}")

        if exclusion_percentage is not None and 0 < exclusion_percentage < 100 and all_steps:
            cut = int(len(all_steps) * (100 - exclusion_percentage) / 100)
            if cut < 1:
                cut = 1
            all_steps = all_steps[:cut]

        return _build_story_graph(all_steps)

    def get_config(self) -> Dict[str, Any]:
        base_docs: List[Dict[str, Any]] = []
        for p in self._base_config_paths:
            try:
                with p.open("r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
                    if isinstance(raw, dict):
                        base_docs.append(cast(Dict[str, Any], raw))
            except Exception as e:
                logger.warning(f"Failed loading base config {p}: {e}")

        overlay_docs: List[Dict[str, Any]] = []
        for p in self._overlay_config_paths:
            try:
                with p.open("r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
                    if isinstance(raw, dict):
                        overlay_docs.append(cast(Dict[str, Any], raw))
            except Exception as e:
                logger.warning(f"Failed loading overlay config {p}: {e}")

        if not base_docs and not overlay_docs:
            return {}

        logger.info("Merging configs...")
        merged = _merge_config_docs(base_docs, overlay_docs)
        # Optional dump support via OVERLAY_DUMP_CONFIG env
        dump_target = os.environ.get("OVERLAY_DUMP_CONFIG", "").strip()
        if dump_target:
            try:
                if dump_target.lower() in {"1", "true", "yes", "stdout"}:
                    yaml.safe_dump(merged, sys.stdout, sort_keys=False, allow_unicode=True)
                else:
                    out_path = Path(dump_target)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with out_path.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)
                    logger.info(f"Dumped merged config to {out_path}")
            except Exception as e:
                logger.warning(f"Failed to dump merged config: {e}")
        return merged

    def get_config_file_for_auto_config(self) -> Optional[str]:
        return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.components.layered_importer <base_domain_path> <overlay_domain_path>")
        sys.exit(1)
    importer = OverlayImporter(base_domain=[sys.argv[1]], overlay_domain=[sys.argv[2]])
    merged_domain = importer.get_domain()
    yaml.safe_dump(cast(Dict[str, Any], merged_domain.as_dict()), sys.stdout, sort_keys=False, allow_unicode=True)
