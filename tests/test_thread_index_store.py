import json
import unittest
from unittest.mock import patch

from src import thread_index_store
from src.thread_index import build_thread_list_from_payload


class _FakePipeline:
    """Reads run at once; writes queue after multi() and apply on execute()."""

    def __init__(self, fake):
        self.fake = fake
        self.queued = []

    def get(self, key):
        return self.fake.store.get(key)

    def multi(self):
        pass

    def set(self, key, value):
        self.queued.append((key, value))

    def execute(self):
        for key, value in self.queued:
            self.fake.store[key] = value


class _FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

    def transaction(self, func, *keys, value_from_callable=False):
        pipe = _FakePipeline(self)
        result = func(pipe)
        pipe.execute()
        return result


class ThreadIndexStoreTests(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeRedis()
        patcher = patch.object(thread_index_store, "_get_client", return_value=self.fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _seed(self, user_sub, payload):
        self.fake.set(thread_index_store._key_for(user_sub), json.dumps(payload))

    def test_get_missing_returns_empty_dict(self):
        self.assertEqual(thread_index_store.get_index_payload("user-1"), {})

    def test_get_reads_the_stored_payload(self):
        payload = {"1": {"id": 1, "name": "First"}}
        self._seed("user-1", payload)
        self.assertEqual(thread_index_store.get_index_payload("user-1"), payload)

    def test_keys_are_scoped_per_user(self):
        self._seed("user-1", {"1": {"id": 1}})
        self._seed("user-2", {"2": {"id": 2}})
        self.assertEqual(thread_index_store.get_index_payload("user-1"), {"1": {"id": 1}})
        self.assertEqual(thread_index_store.get_index_payload("user-2"), {"2": {"id": 2}})

    def test_corrupted_json_returns_empty_dict(self):
        self.fake.set(thread_index_store._key_for("user-1"), "not json")
        self.assertEqual(thread_index_store.get_index_payload("user-1"), {})

    def test_non_dict_json_returns_empty_dict(self):
        self.fake.set(thread_index_store._key_for("user-1"), "[1, 2, 3]")
        self.assertEqual(thread_index_store.get_index_payload("user-1"), {})


class ThreadChangeTests(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeRedis()
        patcher = patch.object(thread_index_store, "_get_client", return_value=self.fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _listed(self, user_sub="user-1"):
        return sorted(build_thread_list_from_payload(thread_index_store.get_index_payload(user_sub)))

    def test_create_allocates_the_next_id_and_names_it_by_default(self):
        first = thread_index_store.create_index_thread("user-1", "")
        second = thread_index_store.create_index_thread("user-1", "  Monthly DTN  ")
        self.assertEqual((first["id"], first["name"]), (1, "Conversation 1"))
        self.assertEqual((second["id"], second["name"]), (2, "Monthly DTN"))
        self.assertEqual(self._listed(), [1, 2])

    def test_create_never_reuses_a_deleted_threads_id(self):
        thread_index_store.create_index_thread("user-1", "")
        self.assertTrue(thread_index_store.delete_index_thread("user-1", 1))
        self.assertEqual(thread_index_store.create_index_thread("user-1", "")["id"], 2)
        self.assertEqual(self._listed(), [2])

    def test_ids_are_per_user(self):
        thread_index_store.create_index_thread("user-1", "")
        self.assertEqual(thread_index_store.create_index_thread("user-2", "")["id"], 1)

    def test_rename_returns_the_renamed_thread(self):
        thread_index_store.create_index_thread("user-1", "")
        renamed = thread_index_store.rename_index_thread("user-1", 1, "Stroke types")
        self.assertEqual((renamed["id"], renamed["name"]), (1, "Stroke types"))

    def test_rename_does_not_bring_back_a_missing_or_deleted_thread(self):
        thread_index_store.create_index_thread("user-1", "")
        thread_index_store.delete_index_thread("user-1", 1)
        self.assertIsNone(thread_index_store.rename_index_thread("user-1", 1, "Back"))
        self.assertIsNone(thread_index_store.rename_index_thread("user-1", 7, "Never"))
        self.assertEqual(self._listed(), [])

    def test_delete_of_a_missing_thread_writes_nothing(self):
        self.assertFalse(thread_index_store.delete_index_thread("user-1", 3))
        self.assertEqual(self.fake.store, {})


if __name__ == "__main__":
    unittest.main()
