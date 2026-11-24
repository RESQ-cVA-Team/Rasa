# pyright: reportUntypedClassDecorator=false, reportUntypedBaseClass=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import re
from typing import Any, Dict, List, Text, cast

from rasa.engine.graph import GraphComponent  # type: ignore
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # type: ignore
from rasa.shared.nlu.training_data.message import Message  # type: ignore

# A standalone year; the digits of an ISO date are part of that date.
_YEAR = re.compile(r"(?<![\w-])(?:19|20)\d{2}(?![\w-])")
# Readings a year cannot have.
_NOT_A_YEAR = frozenset({"age", "nihss"})


def year_dates(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every year in the text is a date: a year outside every date span becomes one, in place of an age or NIHSS read on it."""
    date_spans = [(e.get("start"), e.get("end")) for e in entities if e.get("entity") == "date"]
    out = list(entities)
    for match in _YEAR.finditer(text):
        start, end = match.span()
        if any(isinstance(s, int) and isinstance(e, int) and s <= start and end <= e for s, e in date_spans):
            continue
        out = [e for e in out if not (e.get("entity") in _NOT_A_YEAR and e.get("start") == start and e.get("end") == end)]
        out.append({"entity": "date", "start": start, "end": end, "value": match.group(), "extractor": "YearDates"})
    return out


@DefaultV1Recipe.register(DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False)
class YearDates(GraphComponent):
    """Reads years as dates, so a year never depends on how DIET was trained."""

    @classmethod
    def create(cls, config: Dict[Text, Any], model_storage: Any, resource: Any, execution_context: Any) -> "YearDates":
        return cls()

    def process(self, messages: List[Message]) -> List[Message]:  # type: ignore[override]
        for message in cast(List[Any], messages):
            entities = cast(List[Dict[str, Any]], message.get("entities") or [])
            message.set("entities", year_dates(str(message.get("text") or ""), entities), add_to_output=True)
        return messages
