import unittest

from src.components.year_dates import year_dates


def _entity(entity: str, value: str, start: int, end: int) -> dict:
    return {"entity": entity, "value": value, "start": start, "end": end}


def _dates(text: str, entities: list) -> list:
    return [e["value"] for e in year_dates(text, entities) if e["entity"] == "date"]


class YearDatesTests(unittest.TestCase):
    def test_a_year_read_as_an_age_becomes_a_date(self) -> None:
        text = "What percentage of my patients had a CT scan in 2025?"
        start = text.index("2025")
        out = year_dates(text, [_entity("age", "2025", start, start + 4), _entity("offset", "2025", start, start + 4)])
        self.assertEqual([(e["entity"], e["value"]) for e in out], [("offset", "2025"), ("date", "2025")])

    def test_a_year_with_no_entity_becomes_a_date(self) -> None:
        self.assertEqual(_dates("What is my median DTN in 2024?", []), ["2024"])

    def test_a_year_inside_a_date_stays_part_of_it(self) -> None:
        for text, date in [("Show DTN for Q1 2025", "Q1 2025"), ("Show DTN for March 2025", "March 2025"), ("Show DTN in 2025", "2025")]:
            start = text.index(date)
            self.assertEqual(_dates(text, [_entity("date", date, start, start + len(date))]), [date], text)

    def test_iso_dates_and_other_numbers_are_not_years(self) -> None:
        self.assertEqual(_dates("DTN from 2023-01-01 to 2023-12-31", []), [])
        self.assertEqual(_dates("DTN for patients over 60 with NIHSS 12", [_entity("age", "60", 18, 20)]), [])
        self.assertEqual(_dates("Show 12025 cases", []), [])

    def test_every_year_of_several_is_read(self) -> None:
        self.assertEqual(_dates("Show DTN between 2023 and 2025", []), ["2023", "2025"])
        self.assertEqual(_dates("Ukaž DTN v roce 2025", []), ["2025"])
        self.assertEqual(_dates("Δείξε το DTN το 2025", []), ["2025"])
