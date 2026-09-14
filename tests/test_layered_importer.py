import unittest
from pathlib import Path

from src.components.layered_importer import DEFAULT_SSOT_NLU_ENTITIES, _build_ssot_nlu_doc

_SSOT_DIR = Path(__file__).resolve().parents[1] / "src" / "shared" / "SSOT"


class DefaultSsotNluEntitiesTests(unittest.TestCase):
    def test_chart_type_maps_to_chart_type_yml(self) -> None:
        # Regression guard for the exact bug this test file was added for:
        # chart_type was silently missing from this dict, so ChartType.yml's
        # lookup/synonym data was never generated at train time at all --
        # every chart type still "worked" via hand-authored NLU examples
        # except AREA, which had none, so it was never recognized.
        self.assertEqual(DEFAULT_SSOT_NLU_ENTITIES.get("chart_type"), "ChartType.yml")


class BuildSsotNluDocTests(unittest.TestCase):
    def test_chart_type_lookup_includes_area_and_its_synonyms(self) -> None:
        doc = _build_ssot_nlu_doc(_SSOT_DIR, "en", DEFAULT_SSOT_NLU_ENTITIES)
        self.assertIsNotNone(doc)
        chart_type_lookups = [item for item in doc["nlu"] if item.get("lookup") == "chart_type"]
        self.assertEqual(len(chart_type_lookups), 1)
        examples = chart_type_lookups[0]["examples"]
        self.assertIn("AREA", examples)
        self.assertIn("area chart", examples)
        self.assertIn("area graph", examples)
        self.assertIn("stacked area chart", examples)

    def test_chart_type_lookup_includes_every_other_canonical_chart_type(self) -> None:
        doc = _build_ssot_nlu_doc(_SSOT_DIR, "en", DEFAULT_SSOT_NLU_ENTITIES)
        examples = next(item for item in doc["nlu"] if item.get("lookup") == "chart_type")["examples"]
        for canonical in ("LINE", "BAR", "BOX", "HISTOGRAM", "SCATTER", "PIE", "RADAR", "WATERFALL"):
            self.assertIn(canonical, examples)


if __name__ == "__main__":
    unittest.main()
