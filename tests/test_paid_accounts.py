"""Free Devin credentials leave rotation before any model request is sent."""
import base64
import contextlib
import gzip
import io
import json
from pathlib import Path
import struct
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import devinx
from test_audit_fixes import fake_accounts, isolated_accounts

BODY = {"model": "swe-2-max", "stream": False,
        "messages": [{"role": "user", "content": "hi"}]}
DENIED = ("permission_denied", "This model is no longer available for your "
          "account. Please switch to a different model.")


def jwt(pro):
    claims = {"pro": pro, "exp": time.time() + 3600,
              "teams_tier": "TEAMS_TIER_DEVIN_PRO" if pro else "TEAMS_TIER_DEVIN_FREE"}
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"h.{encoded}.s"


def wire(payload, trailer=False):
    return bytes([2 if trailer else 0]) + struct.pack(">I", len(payload)) + payload


class Backend:
    """Fake both Cognition endpoints; inspect the actual serialized requests."""

    def __init__(self, accounts, paid):
        self.accounts = {account["key"]: account for account in accounts}
        self.paid = set(paid)
        self.auth = []
        self.model = []
        self.classifier_refusals = 0

    def __call__(self, url, data, **kwargs):
        response = mock.Mock(status_code=200)
        if url.endswith(devinx.AUTH_PATH):
            request = devinx.protos()["GetUserJwtRequest"]()
            request.ParseFromString(data)
            account = self.accounts[request.metadata.api_key]
            self.auth.append(account["name"])
            payload = devinx.protos()["GetUserJwtResponse"]()
            payload.user_jwt = jwt(account["name"] in self.paid)
            response.content = payload.SerializeToString()
            return response
        assert url.endswith(devinx.CHAT_PATH), url
        request = devinx.protos()["GetChatMessageRequest"]()
        request.ParseFromString(gzip.decompress(data[5:]))
        account = self.accounts[request.metadata.api_key]
        self.model.append(account["name"])
        if account["name"] not in self.paid:
            code, message = DENIED
        elif self.classifier_refusals:
            self.classifier_refusals -= 1
            code, message = "permission_denied", "input classifier refusal"
        else:
            payload = devinx.protos()["GetChatMessageResponse"]()
            payload.delta_text = "ok"
            payload.stop_reason = 1
            payload.usage.input_tokens = 1
            payload.usage.output_tokens = 1
            response.iter_content.return_value = iter([
                wire(payload.SerializeToString()), wire(b"{}", trailer=True)])
            return response
        response.iter_content.return_value = iter([wire(json.dumps({
            "error": {"code": code, "message": message}}).encode(), trailer=True)])
        return response


class PaidAccountTests(unittest.TestCase):

    def setUp(self):
        self.accounts = fake_accounts("free0", "free1", "free2", "free3", "free4",
                                      "paid0", "paid1", "paid2")
        self.backend = Backend(self.accounts, {"paid0", "paid1", "paid2"})
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(isolated_accounts(self.accounts))
        self.post = self.stack.enter_context(mock.patch.object(
            devinx.SESSION, "post", side_effect=self.backend))
        self.stack.enter_context(mock.patch.object(devinx, "compact_body", side_effect=lambda body: body))
        self.stack.enter_context(mock.patch.object(devinx, "paced", return_value=contextlib.nullcontext()))
        self.stack.enter_context(mock.patch.object(devinx, "RATE_WAIT_BUDGET", 30))
        self.log = io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.log))

    def test_five_free_accounts_do_not_use_the_three_payload_attempts(self):
        with mock.patch.object(devinx, "build_request", wraps=devinx.build_request) as build:
            response, error = devinx.run_swe(dict(BODY), None)
        self.assertIsNone(error)
        self.assertEqual(response["content"], [{"type": "text", "text": "ok"}])
        self.assertEqual(self.backend.model, ["paid0"])
        self.assertEqual(self.backend.auth, ["free0", "free1", "free2", "free3", "free4", "paid0"])
        self.assertEqual([call.args[1] for call in build.call_args_list],
                         [devinx.TOOL_DESC_CAPS[0]] * 6)
        self.assertNotIn("upstream permission_denied", self.log.getvalue())
        self.assertTrue(all(account.get("excluded") for account in self.accounts[:5]))

    def test_streaming_turn_survives_free_account_selection(self):
        target = io.BytesIO()
        result = devinx.run_swe(dict(BODY, stream=True), target)
        self.assertEqual(result, (None, None))
        self.assertEqual(self.backend.model, ["paid0"])
        self.assertNotIn(b"event: error", target.getvalue())
        self.assertIn(b"event: message_stop", target.getvalue())

    def test_excluded_account_does_not_even_refetch_auth(self):
        account = self.accounts[0]
        devinx._check_plan(account, jwt(False))
        request, _ = devinx.build_request(BODY)
        (message, error), = list(devinx.chat_stream(request, account))
        self.assertIsNone(message)
        self.assertTrue(error.startswith("credential_excluded:"))
        self.post.assert_not_called()

    def test_no_paid_account_returns_service_error_not_claude_auth_error(self):
        self.backend.paid.clear()
        outcome = {}
        response, error = devinx.run_swe(dict(BODY), None, outcome=outcome)
        self.assertIsNone(response)
        self.assertEqual(self.backend.model, [])
        self.assertEqual(len(self.backend.auth), 8)
        self.assertEqual(devinx.anthropic_error(error)[:2], ("api_error", 503))
        self.assertEqual(outcome["status"], "503")

    def test_summary_skips_free_accounts_without_model_calls_or_backoff(self):
        with mock.patch.object(devinx.Turn, "sleep") as sleep:
            result = devinx._summarise_once(
                [{"role": "user", "content": "summarize this"}],
                "swe-2-max", "system", None, devinx.Turn())
        self.assertEqual(result, "ok")
        self.assertEqual(self.backend.model, ["paid0"])
        sleep.assert_not_called()

    def test_cached_pro_downgrade_is_rechecked_and_never_retried_on_model(self):
        account = self.accounts[0]
        account.update(jwt=jwt(True), exp=time.time() + 3600)
        response, error = devinx.run_swe(dict(BODY), None)
        self.assertIsNone(error)
        self.assertIsNotNone(response)
        self.assertEqual(self.backend.model, ["free0", "paid0"])
        self.assertTrue(account.get("excluded"))
        self.assertEqual(self.backend.auth.count("free0"), 1)
        for _ in range(20):
            chosen, _ = devinx.claim_account()
            self.assertFalse(chosen.get("excluded"))
            self.assertNotEqual(chosen["name"], "free0")

    def test_non_access_permission_denied_keeps_classifier_retry_budget(self):
        for account in self.accounts[:5]:
            devinx._check_plan(account, jwt(False))
        self.backend.classifier_refusals = 2
        with mock.patch.object(devinx, "build_request", wraps=devinx.build_request) as build:
            response, error = devinx.run_swe(dict(BODY), None)
        self.assertIsNone(error)
        self.assertEqual(self.backend.model, ["paid0"] * 3)
        self.assertEqual([call.args[1] for call in build.call_args_list], list(devinx.TOOL_DESC_CAPS))
        self.assertEqual(self.backend.auth, ["paid0"])

    def test_exclusion_after_auth_but_before_dispatch_prevents_the_model_call(self):
        for purpose in ("turn", "summary"):
            with self.subTest(purpose=purpose):
                account = dict(self.accounts[-1])
                account.update(jwt=jwt(True), exp=time.time() + 3600)
                serialized, release = threading.Event(), threading.Event()
                result, errors = [], []

                def serialize():
                    serialized.set()
                    if not release.wait(3):
                        raise RuntimeError("test did not release serialization")
                    return b"request"

                request = SimpleNamespace(
                    metadata=SimpleNamespace(), chat_message_prompts=[],
                    chat_model_uid="swe-2-max", cascade_id="test",
                    SerializeToString=serialize)

                def work():
                    try:
                        result.extend(devinx.chat_stream(request, account, purpose=purpose))
                    except BaseException as error:
                        errors.append(error)

                worker = threading.Thread(target=work)
                worker.start()
                try:
                    self.assertTrue(serialized.wait(2))
                    devinx._check_plan(account, jwt(False))
                finally:
                    release.set()
                    worker.join(3)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertTrue(result[0][1].startswith("credential_excluded:"))
                self.post.assert_not_called()

    def test_exclusion_and_dispatch_share_the_same_lock(self):
        account = self.accounts[-1]
        account.update(jwt=jwt(True), exp=time.time() + 3600)
        in_post, release = threading.Event(), threading.Event()
        checker_entered, checked = threading.Event(), threading.Event()
        lock = threading.Lock()
        errors = []

        class ObservedLock:
            def __enter__(self):
                if threading.current_thread().name == "check-plan":
                    checker_entered.set()
                lock.acquire()

            def __exit__(self, *args):
                lock.release()

        def post(*args, **kwargs):
            in_post.set()
            if not release.wait(3):
                raise RuntimeError("test did not release dispatch")
            response = mock.Mock(status_code=200)
            response.iter_content.return_value = [wire(b"{}", trailer=True)]
            return response

        def work():
            try:
                request, _ = devinx.build_request(BODY)
                list(devinx.chat_stream(request, account))
            except BaseException as error:
                errors.append(error)

        def exclude():
            try:
                devinx._check_plan(account, jwt(False))
                checked.set()
            except BaseException as error:
                errors.append(error)

        with mock.patch.object(devinx, "_plan_lock_for", return_value=ObservedLock()), \
             mock.patch.object(devinx.SESSION, "post", side_effect=post):
            worker = threading.Thread(target=work)
            checker = threading.Thread(target=exclude, name="check-plan")
            worker.start()
            try:
                self.assertTrue(in_post.wait(2))
                checker.start()
                self.assertTrue(checker_entered.wait(2))
                self.assertFalse(checked.is_set())
                self.assertFalse(account.get("excluded"))
            finally:
                release.set()
                worker.join(3)
                if checker.ident is not None:
                    checker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertFalse(checker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(account.get("excluded"))
        self.assertTrue(checked.is_set())

    def test_dispatch_locks_are_per_account_and_survive_key_rotation(self):
        first, second = self.accounts[-2:]
        lock = devinx._plan_lock_for(first)
        self.assertIsNot(lock, devinx._plan_lock_for(second))
        first["key"] = "replacement-key"
        self.assertIs(lock, devinx._plan_lock_for(first))

    def test_free_accounts_are_not_retried_on_later_turns(self):
        devinx.run_swe(dict(BODY), None)
        self.backend.auth.clear()
        self.backend.model.clear()
        response, error = devinx.run_swe(dict(BODY), None)
        self.assertIsNone(error)
        self.assertIsNotNone(response)
        self.assertTrue(all(name.startswith("paid") for name in self.backend.model))
        self.assertTrue(all(name.startswith("paid") for name in self.backend.auth))


if __name__ == "__main__":
    unittest.main()
