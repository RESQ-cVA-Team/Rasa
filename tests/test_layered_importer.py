import unittest
from pathlib import Path

from src.components import ssot_yaml

from src.components.layered_importer import DEFAULT_SSOT_NLU_ENTITIES, _build_metric_examples, _build_ssot_nlu_doc, _detect_locale, _find_metric_templates

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



class DetectLocaleTests(unittest.TestCase):
    def test_a_locale_build_takes_the_locale_layered_over_en_us(self) -> None:
        paths = [Path("src/locales/en/US/domain"), Path("src/locales/cs/CZ/domain")]
        self.assertEqual(_detect_locale(paths), "cs")

    def test_an_english_build_is_english(self) -> None:
        self.assertEqual(_detect_locale([Path("src/locales/en/US/domain")]), "en")


class LocaleSynonymTests(unittest.TestCase):
    def test_a_czech_model_knows_czech_and_english_metric_names(self) -> None:
        doc = _build_ssot_nlu_doc(_SSOT_DIR, "cs", {"metric": "MetricType.yml"})
        lookup = next(item for item in doc["nlu"] if item.get("lookup") == "metric")["examples"]
        self.assertIn("délka hospitalizace", lookup)
        self.assertIn("length of hospital stay", lookup)

if __name__ == "__main__":
    unittest.main()


class CanonicalSpellingVariantTests(unittest.TestCase):
    def test_metric_lookup_includes_camelcase_and_spaced_spellings_of_canonicals(self) -> None:
        # "AdmissionNihss" typed as a bare reply matched nothing: no synonym has
        # that shape and the lookup only carried ADMISSION_NIHSS verbatim.
        doc = _build_ssot_nlu_doc(_SSOT_DIR, "en", DEFAULT_SSOT_NLU_ENTITIES)
        assert doc is not None
        metric_lookup = next(item for item in doc["nlu"] if item.get("lookup") == "metric")
        self.assertIn("- AdmissionNihss\n", metric_lookup["examples"])
        self.assertIn("- admission nihss\n", metric_lookup["examples"])
        self.assertIn("- ADMISSION_NIHSS\n", metric_lookup["examples"])

    def test_multi_word_synonyms_get_camelcase_spellings_too(self) -> None:
        doc = _build_ssot_nlu_doc(_SSOT_DIR, "en", DEFAULT_SSOT_NLU_ENTITIES)
        assert doc is not None
        metric_lookup = next(item for item in doc["nlu"] if item.get("lookup") == "metric")
        self.assertIn("- DoorToNeedle\n", metric_lookup["examples"])

    def test_single_word_canonicals_get_no_variants(self) -> None:
        doc = _build_ssot_nlu_doc(_SSOT_DIR, "en", DEFAULT_SSOT_NLU_ENTITIES)
        assert doc is not None
        metric_lookup = next(item for item in doc["nlu"] if item.get("lookup") == "metric")
        self.assertEqual(metric_lookup["examples"].count("- DTN\n"), 1)


class MetricExampleTests(unittest.TestCase):
    _ROOT = Path(__file__).resolve().parents[1]

    def test_the_most_specific_layer_has_the_templates(self) -> None:
        paths = [self._ROOT / "src/core/domain", self._ROOT / "src/locales/en/US/domain", self._ROOT / "src/locales/cs/CZ/domain"]
        self.assertEqual(_find_metric_templates(paths), self._ROOT / "src/locales/cs/CZ/templates/metric_examples.yml")

    def test_every_locale_names_every_metric_in_questions_and_chart_requests(self) -> None:
        metrics = len(ssot_yaml.load_ssot_items(_SSOT_DIR / "MetricType.yml"))
        for locale, region in (("en", "US"), ("cs", "CZ"), ("el", "GR")):
            items = _build_metric_examples(_SSOT_DIR, locale, self._ROOT / f"src/locales/{locale}/{region}/templates/metric_examples.yml")
            counts = {item["intent"]: item["examples"].count("\n") for item in items}
            self.assertGreaterEqual(counts.get("ask_question", 0), metrics, locale)
            self.assertGreaterEqual(counts.get("generate_visualization", 0), metrics, locale)

    def test_an_example_annotates_the_metric_name(self) -> None:
        items = _build_metric_examples(_SSOT_DIR, "en", self._ROOT / "src/locales/en/US/templates/metric_examples.yml")
        examples = next(item for item in items if item["intent"] == "ask_question")["examples"]
        self.assertIn("[door to needle](metric)", examples)
