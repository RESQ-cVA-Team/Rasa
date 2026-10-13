import unittest
from pathlib import Path

from src.components.ssot_canonicalizer import _load_ssot_index

_SSOT_DIR = Path(__file__).resolve().parents[1] / "src" / "shared" / "SSOT"


class CanonicalizerSpellingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = _load_ssot_index(_SSOT_DIR / "MetricType.yml")

    def test_camelcase_and_concatenated_canonicals_resolve(self) -> None:
        # Live: "AdmissionNihss" as a bare clarification reply extracted no metric.
        self.assertEqual(self.index.lookup("AdmissionNihss"), "ADMISSION_NIHSS")
        self.assertEqual(self.index.lookup("admissionnihss"), "ADMISSION_NIHSS")
        self.assertEqual(self.index.lookup("ADMISSIONNIHSS"), "ADMISSION_NIHSS")

    def test_camelcase_synonyms_resolve(self) -> None:
        self.assertEqual(self.index.lookup("DoorToNeedle"), "DTN")
        self.assertEqual(self.index.lookup("doortoneedle"), "DTN")

    def test_existing_spellings_still_resolve(self) -> None:
        self.assertEqual(self.index.lookup("ADMISSION_NIHSS"), "ADMISSION_NIHSS")
        self.assertEqual(self.index.lookup("admission nihss"), "ADMISSION_NIHSS")
        self.assertEqual(self.index.lookup("DTN"), "DTN")
        self.assertEqual(self.index.lookup("door to needle"), "DTN")

    def test_unknown_text_still_resolves_to_nothing(self) -> None:
        self.assertIsNone(self.index.lookup("no such metric"))


if __name__ == "__main__":
    unittest.main()
