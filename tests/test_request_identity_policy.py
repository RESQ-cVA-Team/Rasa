import unittest

from src.request_identity_policy import DENY, MALFORMED, OPEN, Decision, required_identity, sender_sub

SUB = "3f2a9c1e-0000-4000-8000-000000000001"


def bound(sub: str) -> Decision:
    return Decision("bound", sub)


class SenderSubTests(unittest.TestCase):
    def test_strips_thread_suffix(self) -> None:
        self.assertEqual(sender_sub(f"{SUB}:thread:12"), SUB)

    def test_leaves_plain_sub_alone(self) -> None:
        self.assertEqual(sender_sub(SUB), SUB)

    def test_only_a_numeric_trailing_thread_id_counts(self) -> None:
        self.assertEqual(sender_sub(f"{SUB}:thread:abc"), f"{SUB}:thread:abc")


class OpenRoutesTests(unittest.TestCase):
    def test_version_is_open_for_read_methods_only(self) -> None:
        self.assertEqual(required_identity("GET", "/version", None), OPEN)
        self.assertEqual(required_identity("HEAD", "/version", None), OPEN)
        self.assertEqual(required_identity("POST", "/version", None), DENY)

    def test_cors_preflight_is_open_on_any_path(self) -> None:
        self.assertEqual(required_identity("OPTIONS", "/model", None), OPEN)
        self.assertEqual(required_identity("options", f"/conversations/{SUB}/tracker", None), OPEN)


class WebhookTests(unittest.TestCase):
    def test_bound_to_the_sender_in_the_body(self) -> None:
        self.assertEqual(
            required_identity("POST", "/webhooks/rest/webhook", f"{SUB}:thread:3"), bound(SUB)
        )

    def test_missing_sender_is_malformed_not_open(self) -> None:
        self.assertEqual(required_identity("POST", "/webhooks/rest/webhook", None), MALFORMED)
        self.assertEqual(required_identity("POST", "/webhooks/rest/webhook", ""), MALFORMED)

    def test_only_post_is_accepted(self) -> None:
        self.assertEqual(required_identity("GET", "/webhooks/rest/webhook", None), DENY)

    def test_channel_health_route_is_denied(self) -> None:
        self.assertEqual(required_identity("GET", "/webhooks/rest/", None), DENY)


class ConversationRoutesTests(unittest.TestCase):
    def test_every_built_in_conversation_route_is_bound_to_the_sender(self) -> None:
        for method, suffix in [
            ("GET", "tracker"),
            ("POST", "tracker/events"),
            ("PUT", "tracker/events"),
            ("GET", "story"),
            ("POST", "execute"),
            ("POST", "trigger_intent"),
            ("POST", "predict"),
            ("POST", "messages"),
            ("DELETE", "tracker"),
        ]:
            with self.subTest(method=method, suffix=suffix):
                self.assertEqual(
                    required_identity(method, f"/conversations/{SUB}:thread:7/{suffix}", None),
                    bound(SUB),
                )

    def test_plain_sub_sender_binds_to_itself(self) -> None:
        self.assertEqual(required_identity("GET", f"/conversations/{SUB}/tracker", None), bound(SUB))

    def test_extra_segments_are_folded_into_the_sender_and_cannot_match_a_real_user(self) -> None:
        decision = required_identity("GET", f"/conversations/{SUB}/tracker/../../other/tracker", None)
        self.assertEqual(decision.kind, "bound")
        self.assertNotEqual(decision.sub, SUB)

    def test_unknown_conversation_subroute_is_denied(self) -> None:
        self.assertEqual(required_identity("GET", f"/conversations/{SUB}/delete-everything", None), DENY)
        self.assertEqual(required_identity("GET", f"/conversations/{SUB}", None), DENY)


class ThreadRoutesTests(unittest.TestCase):
    def test_thread_routes_are_bound_to_the_sub_in_the_path(self) -> None:
        self.assertEqual(required_identity("GET", f"/threads/by-user/{SUB}", None), bound(SUB))
        self.assertEqual(required_identity("GET", f"/threads/by-user/{SUB}/next-id", None), bound(SUB))
        self.assertEqual(required_identity("POST", f"/threads/{SUB}/index-event", None), bound(SUB))
        self.assertEqual(required_identity("DELETE", f"/threads/{SUB}/thread/4", None), bound(SUB))

    def test_malformed_thread_paths_are_denied(self) -> None:
        self.assertEqual(required_identity("GET", "/threads", None), DENY)
        self.assertEqual(required_identity("GET", f"/threads/by-user/{SUB}/extra/segments", None), DENY)
        self.assertEqual(required_identity("DELETE", f"/threads/{SUB}/thread/4/more", None), DENY)


class DeniedByDefaultTests(unittest.TestCase):
    def test_admin_and_introspection_routes_are_denied(self) -> None:
        for method, path in [
            ("GET", "/"),
            ("GET", "/status"),
            ("GET", "/domain"),
            ("POST", "/model/parse"),
            ("POST", "/model/predict"),
            ("POST", "/model/train"),
            ("POST", "/model/test/intents"),
            ("PUT", "/model"),
            ("DELETE", "/model"),
        ]:
            with self.subTest(method=method, path=path):
                self.assertEqual(required_identity(method, path, None), DENY)

    def test_unknown_paths_are_denied(self) -> None:
        self.assertEqual(required_identity("GET", "/socket.io/", None), DENY)
        self.assertEqual(required_identity("GET", "//model/parse", None), DENY)
        self.assertEqual(required_identity("GET", "/version/", None), DENY)


if __name__ == "__main__":
    unittest.main()
