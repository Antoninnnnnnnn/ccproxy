"""The kimi-* route: Claude Code on a Kimi Code model, relayed with a key
ccproxy holds. Hermetic: Kimi is a fake upstream on an ephemeral loopback port,
the keys are fake and live in a temporary data directory.
"""
import http.client
import http.server
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import ccproxy
import kimi
from test_audit_fixes import LiveServer

KEY_A = "sk-kimi-" + "A" * 40
KEY_B = "sk-kimi-" + "B" * 40


def sse(*events):
    return b"".join(f"event:{e['type']}\ndata:{json.dumps(e)}\n\n".encode()
                    for e in events)


THINKING_STREAM = sse(
    {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "content": [],
        "model": "k3.1-pro", "usage": {"input_tokens": 10, "output_tokens": 0}}},
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "thinking_delta", "thinking": "plan"}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "signature_delta", "signature": "SIGPART1"}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "signature_delta", "signature": "SIGPART2"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1,
     "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "text_delta", "text": "ok"}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
     "usage": {"output_tokens": 3}},
    {"type": "message_stop"},
)


class FakeKimi:
    """Records what reached it; `answer(seen)` returns (status, body,
    content type, extra headers)."""

    def __init__(self):
        self.seen = []
        outer = self

        class Echo(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get('content-length') or 0)
                seen = {'path': self.path,
                        'headers': {k.lower(): v for k, v in self.headers.items()},
                        'body': json.loads(self.rfile.read(length) or b'{}')}
                outer.seen.append(seen)
                status, body, ctype, extra = outer.answer(seen)
                self.send_response(status)
                self.send_header('content-type', ctype)
                self.send_header('content-length', str(len(body)))
                for k, v in extra.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.answer = lambda seen: (200, json.dumps({
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": seen['body'].get('model'),
            "content": [{"type": "thinking", "thinking": "plan", "signature": "SIG"},
                        {"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2}}).encode(),
            'application/json', {})
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Echo)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def request(port, path, body, headers=()):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    try:
        conn.request('POST', path, body=json.dumps(body).encode(), headers={
            'content-type': 'application/json', **dict(headers)})
        r = conn.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        conn.close()


BODY = {'model': 'kimi-k3.1-pro', 'max_tokens': 8,
        'messages': [{'role': 'user', 'content': 'hi'}]}


class KimiBase(unittest.TestCase):

    keys = {'main': KEY_A}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = tmp.name
        os.makedirs(kimi.keys_dir(self.data))
        for name, key in self.keys.items():
            with open(os.path.join(kimi.keys_dir(self.data), name + '.key'), 'w') as fh:
                fh.write(key + '\n')
        self.upstream = FakeKimi()
        self.addCleanup(self.upstream.close)
        self.pool = kimi.Pool(self.data)
        env = mock.patch.dict(os.environ, {'CCPROXY_KIMI_KEYS': ''})
        env.start()
        self.addCleanup(env.stop)
        for target, name, value in ((kimi, 'UPSTREAM', self.upstream.url),
                                    (ccproxy, 'KIMI_POOL', self.pool),
                                    (kimi, 'refresh_catalog', lambda *a, **k: False)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.live = LiveServer()
        self.addCleanup(self.live.close)
        self.stdout = mock.patch('sys.stdout', new_callable=io.StringIO)
        self.stdout.start()
        self.addCleanup(self.stdout.stop)

    def post(self, body=BODY, path='/v1/messages', headers=()):
        return request(self.live.port, path, body, headers)


class KimiRouteTests(KimiBase):

    def test_the_kimi_key_replaces_the_claude_credential(self):
        status, _, _ = self.post(headers={
            'Authorization': 'Bearer claude-secret', 'X-Api-Key': 'sk-ant-secret',
            'Cookie': 'session=secret', 'anthropic-beta': 'oauth-2025-04-20'})
        self.assertEqual(status, 200)
        (seen,) = self.upstream.seen
        self.assertEqual(seen['path'], '/v1/messages')
        self.assertEqual(seen['headers']['x-api-key'], KEY_A)
        self.assertNotIn('authorization', seen['headers'])
        self.assertNotIn('cookie', seen['headers'])
        self.assertEqual(seen['body']['model'], 'k3.1-pro')

    def test_an_unknown_model_is_refused_not_silently_defaulted(self):
        # Kimi answers 200 for any model name, running its default instead.
        status, _, raw = self.post(dict(BODY, model='kimi-k9-typo'))
        self.assertEqual(status, 400)
        self.assertIn('kimi-k3.1-pro', json.loads(raw)['error']['message'])
        self.assertEqual(self.upstream.seen, [])

    def test_no_account_says_how_to_add_one(self):
        os.remove(os.path.join(kimi.keys_dir(self.data), 'main.key'))
        self.pool.accounts = []
        status, _, raw = self.post()
        self.assertEqual(status, 503)
        self.assertIn('--kimi-add', json.loads(raw)['error']['message'])

    def test_browser_requests_are_refused(self):
        for headers in ({'Origin': 'https://evil.example'},
                        {'Host': 'evil.example:8316'}):
            with self.subTest(headers=headers):
                status, _, _ = self.post(headers=headers)
                self.assertEqual(status, 403)
        self.assertEqual(self.upstream.seen, [])

    def test_kimi_signatures_are_tagged_in_a_stream(self):
        self.upstream.answer = lambda seen: (200, THINKING_STREAM,
                                             'text/event-stream', {})
        status, _, raw = self.post(dict(BODY, stream=True))
        self.assertEqual(status, 200)
        sigs = [json.loads(line[5:])['delta']['signature']
                for line in raw.decode().splitlines()
                if line.startswith('data:') and 'signature_delta' in line]
        # The client concatenates the pieces: one tag, on the first.
        self.assertEqual(sigs, ['kimi:SIGPART1', 'SIGPART2'])
        self.assertIn(b'message_stop', raw)

    def test_kimi_signatures_are_tagged_in_a_whole_message(self):
        status, headers, raw = self.post()
        self.assertEqual(status, 200)
        self.assertEqual(int(headers['content-length']), len(raw))
        message = json.loads(raw)
        self.assertEqual(message['content'][0]['signature'], 'kimi:SIG')

    def test_tagged_thinking_goes_back_to_kimi_untagged(self):
        body = dict(BODY, messages=[
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'p', 'signature': 'kimi:SIG'},
                {'type': 'text', 'text': 'hello'}]},
            {'role': 'user', 'content': 'again'}])
        self.post(body)
        sent = self.upstream.seen[0]['body']['messages'][1]['content'][0]
        self.assertEqual(sent['signature'], 'SIG')

    def test_count_tokens_is_relayed_with_the_upstream_model(self):
        self.upstream.answer = lambda seen: (
            200, b'{"input_tokens": 42}', 'application/json', {})
        status, _, raw = self.post(path='/v1/messages/count_tokens')
        self.assertEqual((status, json.loads(raw)), (200, {'input_tokens': 42}))
        self.assertEqual(self.upstream.seen[0]['path'], '/v1/messages/count_tokens')
        self.assertEqual(self.upstream.seen[0]['body']['model'], 'k3.1-pro')

    def test_the_advisor_server_tool_is_taken_out(self):
        body = dict(BODY, tools=[{'type': 'advisor_20260301', 'name': 'advisor'},
                                 {'name': 'Read', 'input_schema': {}}])
        self.post(body)
        self.assertEqual([t['name'] for t in self.upstream.seen[0]['body']['tools']],
                         ['Read'])

    def test_a_context_refusal_carries_both_figures_for_the_main_session(self):
        self.upstream.answer = lambda seen: (400, json.dumps({
            'type': 'error', 'error': {'type': 'invalid_request_error',
                                       'message': 'Your request exceeded k3.1-pro '
                                                  'model token limit: 1048576'}}).encode(),
            'application/json', {})
        status, _, raw = self.post()
        self.assertEqual(status, 400)
        self.assertRegex(json.loads(raw)['error']['message'],
                         r'^prompt is too long: \d+ tokens > 1048576 maximum$')

    def test_other_errors_pass_through(self):
        payload = b'{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}'
        self.upstream.answer = lambda seen: (400, payload, 'application/json', {})
        status, _, raw = self.post()
        self.assertEqual((status, raw), (400, payload))
        self.assertEqual(len(self.upstream.seen), 1)


class KimiAccountTests(KimiBase):

    keys = {'one': KEY_A, 'two': KEY_B}

    def test_turns_go_round_robin(self):
        self.post()
        self.post()
        used = [s['headers']['x-api-key'] for s in self.upstream.seen]
        self.assertEqual(sorted(used), sorted([KEY_A, KEY_B]))

    def test_a_spent_account_hands_the_turn_to_the_next(self):
        def answer(seen):
            if seen['headers']['x-api-key'] == KEY_A:
                return (429, b'{"type":"error","error":{"type":"rate_limit_error",'
                             b'"message":"rate limit"}}', 'application/json',
                        {'retry-after': '120'})
            return (200, b'{"type":"message","content":[],"usage":{}}',
                    'application/json', {})
        self.upstream.answer = answer
        for _ in range(2):
            status, _, _ = self.post()
            self.assertEqual(status, 200)
        one = next(a for a in self.pool.accounts if a['key'] == KEY_A)
        self.assertGreater(one['blocked_until'], 0)
        # Once blocked, the spent account is not tried again.
        keys = [s['headers']['x-api-key'] for s in self.upstream.seen]
        self.assertEqual(keys.count(KEY_A), 1)

    def test_every_account_spent_is_a_rate_limit_with_retry_after(self):
        self.upstream.answer = lambda seen: (
            429, b'{"type":"error","error":{"type":"rate_limit_error","message":"x"}}',
            'application/json', {'retry-after': '300'})
        status, headers, raw = self.post()
        self.assertEqual(status, 429)
        self.assertEqual(json.loads(raw)['error']['type'], 'rate_limit_error')
        self.assertGreater(int(headers['retry-after']), 0)
        self.assertEqual(len(self.upstream.seen), 2)

    def test_a_refused_key_is_set_aside(self):
        def answer(seen):
            if seen['headers']['x-api-key'] == KEY_A:
                return (401, b'{"error":{"type":"authentication_error",'
                             b'"message":"Invalid Authentication"}}',
                        'application/json', {})
            return (200, b'{"type":"message","content":[],"usage":{}}',
                    'application/json', {})
        self.upstream.answer = answer
        for _ in range(3):
            status, _, _ = self.post()
            self.assertEqual(status, 200)
        keys = [s['headers']['x-api-key'] for s in self.upstream.seen]
        self.assertEqual(keys.count(KEY_A), 1)
        one = next(a for a in self.pool.accounts if a['key'] == KEY_A)
        self.assertIsNotNone(one['invalid'])


class KimiCompactionTests(KimiBase):

    def test_a_subagent_context_refusal_is_retried_compacted_harder(self):
        calls = []

        def answer(seen):
            calls.append(seen)
            if len(calls) == 1:
                return (400, b'{"type":"error","error":{"type":"invalid_request_error",'
                             b'"message":"Your request exceeded k3.1-pro model token '
                             b'limit: 1048576"}}', 'application/json', {})
            return (200, b'{"type":"message","content":[],"usage":{}}',
                    'application/json', {})
        self.upstream.answer = answer
        forced = []

        def spy(handler, body, force=False, committed=False, route='gpt'):
            forced.append((force, route))
            return body, committed
        with mock.patch.object(ccproxy.Handler, 'compact_gpt', spy):
            status, _, _ = self.post(headers={'x-claude-code-agent-id': 'agent-1'})
        self.assertEqual(status, 200)
        self.assertEqual(forced, [(False, 'kimi'), (True, 'kimi')])

    def test_compaction_threshold_follows_each_window(self):
        with mock.patch.object(kimi, 'COMPACT_AT', None):
            self.assertEqual(kimi.compact_at('kimi-k3.1-pro'), 1048576 * 9 // 10)
            self.assertEqual(kimi.compact_at('kimi-k3-256k'), 262144 * 9 // 10)


class KimiShapingTests(unittest.TestCase):

    def test_effort_maps_onto_kimis_three_levels(self):
        for given, sent in (('low', 'low'), ('medium', 'high'), ('high', 'high'),
                            ('xhigh', 'max'), ('max', 'max')):
            out = kimi.prepare({'model': 'kimi-k3', 'output_config': {'effort': given},
                                'messages': []}, 'k3')
            self.assertEqual(out['output_config']['effort'], sent)

    def test_disabled_thinking_is_dropped(self):
        # K3.1 Pro: "does not support disabled thinking mode".
        out = kimi.prepare({'thinking': {'type': 'disabled'}, 'messages': []}, 'k3')
        self.assertNotIn('thinking', out)
        out = kimi.prepare({'thinking': {'type': 'adaptive'}, 'messages': []}, 'k3')
        self.assertEqual(out['thinking'], {'type': 'adaptive'})

    def test_tool_addition_blocks_are_dropped(self):
        # Kimi refuses a request carrying one ("Invalid request Error").
        out = kimi.prepare({'messages': [
            {'role': 'user', 'content': 'hi'},
            {'role': 'system', 'content': [
                {'type': 'text', 'text': '# Environment'},
                {'type': 'tool_addition',
                 'tool': {'type': 'tool_reference', 'name': 'advisor'}}]},
            {'role': 'system', 'content': [
                {'type': 'tool_addition',
                 'tool': {'type': 'tool_reference', 'name': 'Read'}}]}]}, 'k3')
        self.assertEqual(out['messages'], [
            {'role': 'user', 'content': 'hi'},
            {'role': 'system', 'content': [{'type': 'text', 'text': '# Environment'}]}])

    def test_prepare_leaves_the_clients_body_alone(self):
        body = {'model': 'kimi-k3', 'messages': [{'role': 'assistant', 'content': [
            {'type': 'thinking', 'thinking': 'p', 'signature': 'kimi:S'}]}]}
        kimi.prepare(body, 'k3')
        self.assertEqual(body['model'], 'kimi-k3')
        self.assertEqual(body['messages'][0]['content'][0]['signature'], 'kimi:S')

    def test_claude_route_drops_kimi_thinking(self):
        body = {'thinking': {'type': 'adaptive'}, 'messages': [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'p', 'signature': 'kimi:S'},
                {'type': 'text', 'text': 'done'}]}]}
        self.assertTrue(ccproxy.strip_foreign_thinking(body))
        self.assertEqual(body['messages'][1]['content'],
                         [{'type': 'text', 'text': 'done'}])

    def test_overflow_and_account_refusals_are_told_apart(self):
        text = 'Your request exceeded k3-256k model token limit: 262144'
        self.assertEqual(kimi.overflow(400, text), 262144)
        self.assertIsNone(kimi.account_refusal(400, text))
        self.assertIsNone(kimi.overflow(400, 'bad request'))
        self.assertEqual(kimi.account_refusal(401, 'Invalid Authentication'), 'auth')
        self.assertEqual(kimi.account_refusal(429, ''), 'quota')
        self.assertIsNone(kimi.account_refusal(400, 'messages: field required'))

    def test_usage_is_read_from_the_usages_endpoint_shape(self):
        info = kimi.parse_usage({'usages': {
            'limit_5h': {'used_ratio': 0.25, 'reset_time': '2026-11-05T18:33:31Z'},
            'limit_month_total': {'used_ratio': 0.0626,
                                  'reset_time': '2026-11-25T00:00:00Z'}}})
        self.assertEqual(info['5h'][0], 0.25)
        self.assertEqual(info['month'][0], 0.0626)
        self.assertIsNotNone(info['5h'][1])


class KimiAccountFileTests(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = tmp.name
        env = mock.patch.dict(os.environ, {'CCPROXY_KIMI_KEYS': ''})
        env.start()
        self.addCleanup(env.stop)
        usage = mock.patch.object(kimi, '_get_json', return_value={'usages': {}})
        self.get = usage.start()
        self.addCleanup(usage.stop)

    def test_add_saves_a_private_file_and_names_it(self):
        out = io.StringIO()
        self.assertEqual(kimi.add(self.data, key=KEY_A, out=out), 0)
        path = os.path.join(kimi.keys_dir(self.data), 'kimi1.key')
        self.assertEqual(open(path).read().strip(), KEY_A)
        if os.name == 'posix':
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertNotIn(KEY_A, out.getvalue())
        # The same key again is the same account, not a second one.
        kimi.add(self.data, key=KEY_A, out=out)
        kimi.add(self.data, key=KEY_B, out=out)
        self.assertEqual(sorted(a['name'] for a in kimi.load_accounts(self.data)),
                         ['kimi1', 'kimi2'])

    def test_a_key_kimi_refuses_is_not_saved(self):
        import urllib.error
        self.get.side_effect = urllib.error.HTTPError('u', 401, 'no', {}, None)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(kimi.add(self.data, key=KEY_A, out=io.StringIO()), 1)
        self.assertFalse(kimi.configured(self.data))

    def test_env_keys_replace_the_files(self):
        kimi.add(self.data, 'file', key=KEY_A, out=io.StringIO())
        with mock.patch.dict(os.environ, {'CCPROXY_KIMI_KEYS': KEY_B}):
            self.assertEqual([a['key'] for a in kimi.load_accounts(self.data)], [KEY_B])


class KimiLauncherTests(unittest.TestCase):

    def setUp(self):
        import launcher
        self.launcher = launcher
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = mock.patch.dict(os.environ, {'CCPROXY_DATA': tmp.name,
                                           'CCPROXY_KIMI_KEYS': ''})
        env.start()
        self.addCleanup(env.stop)
        self.data = tmp.name

    def test_account_commands_behind_a_wrappers_flags(self):
        flag = self.launcher._kimi_flag
        self.assertEqual(flag(['--kimi-add', 'work']), ('--kimi-add', ['work']))
        self.assertEqual(flag(['--devin', '--dangerously-skip-permissions',
                               '--kimi-list']), ('--kimi-list', []))
        self.assertIsNone(flag(['-p', 'hello', '--kimi-add']))
        self.assertIsNone(flag(['--', '--kimi-add']))

    def settings(self):
        return json.loads(self.launcher.picker_settings([])[1])

    def test_nothing_kimi_without_an_account(self):
        rows = [r['model'] for r in self.settings()['modelPicker']['options']]
        self.assertFalse([m for m in rows if m.startswith('kimi-')])
        self.assertFalse([a for a in self.launcher.packaged_agents()
                          if a.startswith('kimi-')])

    def test_rows_and_agents_with_an_account(self):
        with mock.patch.dict(os.environ, {'CCPROXY_KIMI_KEYS': KEY_A}):
            settings = self.settings()
            agents = self.launcher.packaged_agents()
        rows = {r['model']: r for r in settings['modelPicker']['options']}
        self.assertIn('kimi-k3.1-pro', rows)
        for model, row in rows.items():
            if model.startswith('kimi-'):
                self.assertEqual(row['behavesAs'], 'claude-opus-5-5')
                self.assertEqual(
                    settings['modelSettings'][model]['autoCompactWindow'],
                    kimi.context_tokens(model) * 9 // 10 + 33000)
                self.assertIsNotNone(kimi.upstream_model(model))
        for short in ('pro', 'k3'):
            for effort in ('low', 'high', 'max'):
                agent = agents[f'kimi-{short}-{effort}']
                self.assertEqual(agent['effort'], effort)
                self.assertIsNotNone(kimi.upstream_model(agent['model']))
                self.assertEqual(agent['disallowedTools'], ['mcp__*'])


if __name__ == '__main__':
    unittest.main()
