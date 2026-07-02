import unittest

from src.thread_index import (
    INDEX_MARKER,
    apply_index_action,
    build_thread_list_from_payload,
    extract_index_payload_from_events,
    next_thread_id_from_payload,
    serialize_index_payload,
)


class _Event:
    def __init__(self, text):
        self.text = text


class ThreadIndexTests(unittest.TestCase):
    def test_extract_payload_uses_latest_marker(self):
        events = [
            _Event("plain"),
            _Event(f"{INDEX_MARKER}{{\"1\":{{\"id\":1,\"name\":\"A\",\"action\":\"create\"}}}}"),
            _Event(f"{INDEX_MARKER}{{\"1\":{{\"id\":1,\"name\":\"B\",\"action\":\"rename\"}}}}"),
        ]
        payload = extract_index_payload_from_events(events)
        self.assertEqual(payload["1"]["name"], "B")

    def test_build_thread_list_filters_deleted_entries(self):
        payload = {
            "1": {"id": 1, "name": "One", "action": "create", "created_at": "c1", "timestamp": "t1"},
            "2": {"id": 2, "name": "Two", "action": "delete", "created_at": "c2", "timestamp": "t2"},
        }
        threads = build_thread_list_from_payload(payload)
        self.assertIn(1, threads)
        self.assertNotIn(2, threads)

    def test_apply_index_action_and_next_id(self):
        payload = {}
        payload = apply_index_action(payload, 1, "create", "First")
        payload = apply_index_action(payload, 1, "rename", "Renamed")
        payload = apply_index_action(payload, 2, "create", "Second")

        self.assertEqual(payload["1"]["name"], "Renamed")
        self.assertEqual(next_thread_id_from_payload(payload), 3)

    def test_serialize_payload_prefixes_marker(self):
        text = serialize_index_payload({"1": {"id": 1}})
        self.assertTrue(text.startswith(INDEX_MARKER))


if __name__ == "__main__":
    unittest.main()