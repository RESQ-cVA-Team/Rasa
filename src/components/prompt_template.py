"""Pure helper for making an LLM component's system prompt admin-editable
(via config.yml -> CVaLab's pipeline editor) without an edit that drops or
breaks a placeholder silently degrading or crashing the component.
Rasa-independent, so it's testable on its own -- same reasoning as
intent_matching.py and llm_nlu_parsing.py.

Shared by every LLM component with an editable prompt (llm_intent_fallback.py,
llm_nlu_classifier.py on the experimental branch) rather than each hand-rolling
its own "did the edit break the placeholders" check.
"""
from __future__ import annotations

from typing import List, Optional, Tuple


def render_system_prompt(
    template: str,
    default_template: str,
    required_placeholders: List[str],
    **kwargs: str,
) -> Tuple[str, Optional[str]]:
    """Fills `template` with kwargs, falling back to `default_template` (also
    filled with the same kwargs) if `template` is missing a placeholder named
    in `required_placeholders`, or fails to render at all (e.g. references a
    name not present in kwargs).

    Returns (rendered_prompt, warning_or_None) -- callers should log the
    warning, not just discard it: an admin's edit silently degrading the
    prompt's real grounding (or being silently reverted) should be visible,
    not a quiet no-op. The component still runs with a working prompt
    either way -- a bad edit is a warning, never a crash or a broken
    deployment."""
    missing = [p for p in required_placeholders if p not in template]
    if missing:
        return (
            default_template.format(**kwargs),
            f"system_prompt_template is missing required placeholder(s) {missing}; using the default template instead.",
        )
    try:
        return template.format(**kwargs), None
    except (KeyError, IndexError) as exc:
        return (
            default_template.format(**kwargs),
            f"system_prompt_template failed to render ({exc!r}); using the default template instead.",
        )
