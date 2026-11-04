#!/usr/bin/env python3
"""Devin / SWE-2 access is off unless DEVINX_SWE=1.

The service refuses swe-* models, neither catalog lists them, and the launcher
offers no swe2-* agent, no SWE-2 /model entry and no Codex SWE role. Claude and
GPT are untouched. Hermetic: no network, no real service, no Devin credential.

    .venv/bin/python -m unittest tests.test_swe_disabled -v
"""
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import devinx
import launcher
import runtime_support
from test_audit_fixes import LiveServer


def off():
    """The environment of a user who never set DEVINX_SWE."""
    env = {k: v for k, v in os.environ.items() if k != "DEVINX_SWE"}
    return mock.patch.dict(os.environ, env, clear=True)


def on():
    return mock.patch.dict(os.environ, {"DEVINX_SWE": "1"})


def post(port, path, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("POST", path, body=json.dumps(body).encode(), headers={
            "content-type": "application/json",
            "authorization": "Bearer claude-secret",
            "anthropic-version": "2023-06-01"})
        r = conn.getresponse()
        return r.status, json.loads(r.read() or b"{}")
    finally:
        conn.close()


class SwitchTests(unittest.TestCase):
    def test_off_unless_exactly_one(self):
        for value, expected in ((None, False), ("", False), ("0", False),
                                ("true", False), ("1", True)):
            with self.subTest(value=value):
                env = {k: v for k, v in os.environ.items() if k != "DEVINX_SWE"}
                if value is not None:
                    env["DEVINX_SWE"] = value
                with mock.patch.dict(os.environ, env, clear=True):
                    self.assertIs(runtime_support.swe_enabled(), expected)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.live = LiveServer()
        self.addCleanup(self.live.close)

    def test_swe_models_are_refused_before_any_credential_is_read(self):
        # Any of these being called would mean a Devin quota could be spent.
        boom = AssertionError("a Devin credential was touched")
        with off(), \
                mock.patch.object(devinx, "accounts", side_effect=boom), \
                mock.patch.object(devinx, "claim_account", side_effect=boom), \
                mock.patch.object(devinx, "get_jwt", side_effect=boom), \
                mock.patch.object(devinx, "_first_usable", side_effect=boom):
            for model in ("swe-2", "swe-2-max", "swe-2-high", "swe-2-medium"):
                for path in ("/v1/messages", "/v1/messages/count_tokens",
                             "/v1/responses"):
                    with self.subTest(model=model, path=path):
                        status, body = post(self.live.port, path, {
                            "model": model, "max_tokens": 8,
                            "messages": [{"role": "user", "content": "hi"}]})
                        self.assertEqual(status, 403)
                        self.assertEqual(body["error"]["type"],
                                         "permission_error")
                        self.assertIn("disabled", body["error"]["message"])

    def test_v1_models_has_no_swe_row_but_keeps_gpt(self):
        with off():
            status, body = self.live.get("/v1/models", {})
        self.assertEqual(status, 200)
        ids = [m["id"] for m in body["data"]]
        self.assertFalse([i for i in ids if i.startswith("swe-")], ids)
        for mid in ("gpt-6.1-sol", "gpt-6-astra", "gpt-6-luna"):
            self.assertIn(mid, ids)
        self.assertEqual(body["first_id"], ids[0])

    def test_codex_catalog_has_no_swe_row(self):
        upstream = mock.Mock(status_code=200)
        upstream.json.return_value = {"models": [{"slug": "gpt-test"}]}
        with off(), mock.patch.object(devinx.SESSION, "get",
                                      return_value=upstream):
            status, body = self.live.get(
                "/v1/models?client_version=1.0",
                {"Authorization": "Bearer codex-secret"})
        self.assertEqual(status, 200)
        slugs = [m["slug"] for m in body["models"]]
        self.assertEqual(slugs, ["gpt-test"])

    def test_switched_on_the_catalog_lists_swe_first_again(self):
        with on():
            status, body = self.live.get("/v1/models", {})
        self.assertEqual(status, 200)
        self.assertEqual(body["first_id"], "swe-2")


class LauncherTests(unittest.TestCase):
    def test_no_swe_agent_is_offered_but_claude_and_gpt_are(self):
        with off():
            agents = launcher.packaged_agents()
        self.assertFalse([n for n in agents if n.startswith("swe2-")], agents)
        self.assertIn("claude-reviewer", agents)
        self.assertIn("gpt-sol-high", agents)
        self.assertFalse(
            [n for n, a in agents.items()
             if str(a.get("model", "")).startswith("swe-")])

    def test_swe_agents_return_when_switched_on(self):
        with on():
            agents = launcher.packaged_agents()
        self.assertIn("swe2-worker", agents)

    def test_codex_gets_no_swe_role_or_default_subagent_model(self):
        with off():
            joined = " ".join(launcher.codex_args(True))
        self.assertNotIn("swe2-", joined)
        self.assertNotIn("swe-2", joined)
        with on():
            joined = " ".join(launcher.codex_args(True))
        self.assertIn("swe2-explorer", joined)
        self.assertIn('agents.default_subagent_model="swe-2-max"', joined)

    def test_no_swe_model_entry_in_the_launched_environment(self):
        # launcher.ENV is built at import time, so ask a fresh interpreter.
        code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
                "import json, launcher; "
                "print(json.dumps(sorted(k for k in launcher.ENV "
                "if k.startswith('ANTHROPIC_CUSTOM_MODEL_OPTION'))))")
        for value, expected in ((None, []),
                                ("1", ["ANTHROPIC_CUSTOM_MODEL_OPTION",
                                       "ANTHROPIC_CUSTOM_MODEL_OPTION_"
                                       "SUPPORTED_CAPABILITIES"])):
            with self.subTest(DEVINX_SWE=value):
                env = {k: v for k, v in os.environ.items()
                       if k != "DEVINX_SWE"}
                if value:
                    env["DEVINX_SWE"] = value
                out = subprocess.run([sys.executable, "-I", "-c", code],
                                     env=env, capture_output=True, text=True,
                                     cwd=str(ROOT), check=True)
                self.assertEqual(json.loads(out.stdout), expected)

    def test_a_catalog_without_swe_still_identifies_devinx(self):
        catalog = {"data": [{"id": "gpt-6-luna", "owned_by": "devinx"}]}
        reply = mock.MagicMock()
        reply.__enter__.return_value.read.return_value = json.dumps(
            catalog).encode()
        sock = mock.MagicMock()
        sock.__enter__.return_value.connect_ex.return_value = 0
        with mock.patch.object(socket, "socket", return_value=sock), \
                mock.patch.object(launcher._opener, "open", return_value=reply):
            self.assertTrue(launcher.listening())

    def test_a_foreign_catalog_is_still_foreign(self):
        catalog = {"data": [{"id": "gpt-4", "owned_by": "openai"}]}
        reply = mock.MagicMock()
        reply.__enter__.return_value.read.return_value = json.dumps(
            catalog).encode()
        sock = mock.MagicMock()
        sock.__enter__.return_value.connect_ex.return_value = 0
        with mock.patch.object(socket, "socket", return_value=sock), \
                mock.patch.object(launcher._opener, "open", return_value=reply):
            self.assertFalse(launcher.listening())


class InstallerTests(unittest.TestCase):
    def test_no_credential_is_asked_for_while_access_is_off(self):
        import install
        with off(), mock.patch.object(install, "say") as say:
            self.assertFalse(install.check_credential())
        levels = [call.args[0] for call in say.call_args_list]
        self.assertNotIn(install.WARN, levels)
        self.assertIn("disabled", say.call_args.args[1])

if __name__ == "__main__":
    unittest.main()
