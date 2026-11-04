"""Stream relay tests: SSE framing, failure accounting, the gpt-* retry.

Hermetic like the rest of the suite: the upstream is a mock or a scripted
HTTP server on an ephemeral loopback port, the handler a bare instance whose
wfile is a BytesIO, and nothing reaches a real service.
"""
import contextlib
import gzip
import http.client
import http.server
import io
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock
import zlib

import urllib3.exceptions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import devinx
from test_audit_fixes import LiveServer


def event(kind, **fields):
    return devinx.sse_bytes({"type": kind, **fields})


MESSAGE_START = event("message_start", message={
    "id": "msg_1", "type": "message", "role": "assistant",
    "usage": {"input_tokens": 12}})
PING = event("ping")
COMMENT = b": keep-alive\n\n"
CONTENT_START = event("content_block_start", index=0,
                      content_block={"type": "text", "text": ""})
THINKING_START = event("content_block_start", index=0,
                       content_block={"type": "thinking", "thinking": ""})
TOOL_START = event("content_block_start", index=0,
                   content_block={"type": "tool_use", "id": "toolu_1",
                                  "name": "Bash", "input": {}})
TEXT_DELTA = event("content_block_delta", index=0,
                   delta={"type": "text_delta", "text": "hi"})
BLOCK_STOP = event("content_block_stop", index=0)
MESSAGE_DELTA = event("message_delta", delta={"stop_reason": "end_turn"},
                      usage={"output_tokens": 4})
MESSAGE_STOP = event("message_stop")
OK_STREAM = [MESSAGE_START, CONTENT_START, TEXT_DELTA, BLOCK_STOP,
             MESSAGE_DELTA, MESSAGE_STOP]


def error_event(kind, message="upstream failed", code=None):
    err = {"type": kind, "message": message}
    if code is not None:
        err["code"] = code
    return devinx.sse_bytes({"type": "error", "error": err})


def chop(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]


def upstream(chunks, status=200, headers=None, exc=None):
    """A requests.Response stand-in: raw.stream yields chunks, then raises exc.

    raw.read answers with the same bytes for the error paths that consume the
    body instead of streaming it.
    """
    response = mock.Mock(status_code=status)
    response.headers = {"content-type": "text/event-stream", **(headers or {})}
    response.raw = mock.Mock(spec=["stream", "read"])

    def stream(*args, **kwargs):
        yield from chunks
        if exc is not None:
            raise exc
    response.raw.stream.side_effect = stream
    response.raw.read.return_value = b"".join(
        c for c in chunks if isinstance(c, bytes))
    return response


def bare_handler(headers=None):
    """A Handler with no socket; whatever it writes lands in wfile."""
    handler = devinx.Handler.__new__(devinx.Handler)
    handler.wfile = io.BytesIO()
    handler.command = "POST"
    handler.path = "/v1/messages"
    handler.requestline = "POST /v1/messages HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.headers = dict(headers or {})
    return handler


def split_response(payload):
    """Status line, headers dict and body out of what the handler wrote."""
    head, _, body = payload.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return lines[0], headers, body


class RelayHarness:
    """Drive handler.relay against scripted responses, capturing the log."""

    def setUp(self):
        # A dump path left over from a real investigation must never make a
        # test write the request or the stream to disk.
        patcher = mock.patch.dict(os.environ, {"DEVINX_DUMP": ""})
        patcher.start()
        self.addCleanup(patcher.stop)

    def relay(self, responses, label="claude", headers=None, raw=b'{"model":"m"}',
              **kwargs):
        handler = bare_handler(headers)
        log = io.StringIO()
        with mock.patch.object(devinx.SESSION, "request",
                               side_effect=list(responses)) as request, \
             contextlib.redirect_stdout(log):
            handler.relay(raw, "model-x", "http://upstream/v1/messages", label,
                          **kwargs)
        status, response_headers, body = split_response(handler.wfile.getvalue())
        return request, log.getvalue(), status, response_headers, body


class SseFrameTests(unittest.TestCase):
    """sse_frames/sse_events: boundaries, encodings and the frame budget."""

    def test_separator_split_across_chunks(self):
        # The \r\n\r\n arriving a byte at a time is still one boundary.
        raw = (b"event: ping\r\ndata: {\"type\":\"ping\"}\r\n\r\n"
               b"event: x\ndata: {\"type\":\"x\"}\n\n")
        frames = list(devinx.sse_frames(chop(raw, 1)))
        self.assertEqual(len(frames), 2)
        self.assertEqual(devinx.parse_sse_frame(frames[0]),
                         ("ping", {"type": "ping"}))
        self.assertEqual(devinx.parse_sse_frame(frames[1]),
                         ("x", {"type": "x"}))

    def test_multi_data_lines_form_one_json(self):
        frame = b"data: {\"a\":\ndata: 1}\n\n"
        self.assertEqual(devinx.parse_sse_frame(frame), (None, {"a": 1}))

    def test_comment_frames_carry_no_event(self):
        self.assertEqual(list(devinx.sse_events([b": keep-alive\n\n"])),
                         [(None, None)])

    def test_utf8_split_mid_character_reassembles(self):
        frame = devinx.sse_bytes({"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta",
                                            "text": "héllo ✓"}})
        events = list(devinx.sse_events(chop(frame, 3)))
        self.assertEqual(len(events), 1)
        name, data = events[0]
        self.assertEqual(data["delta"]["text"], "héllo ✓")

    def test_oversized_frame_is_refused(self):
        with mock.patch.object(devinx, "MAX_FRAME_BYTES", 64):
            with self.assertRaises(ValueError):
                list(devinx.sse_frames([b"data: " + b"x" * 100]))

    def test_malformed_data_parses_to_none_not_a_crash(self):
        name, data = devinx.parse_sse_frame(b"data: {not json\n\n")
        self.assertIsNone(data)


class UsageSnifferTests(unittest.TestCase):

    def test_usage_from_split_lines(self):
        sniff = devinx.UsageSniffer()
        raw = MESSAGE_START + MESSAGE_DELTA
        for piece in chop(raw, 7):
            sniff.feed(piece)
        self.assertEqual(sniff.usage["input_tokens"], 12)
        self.assertEqual(sniff.usage["output_tokens"], 4)
        self.assertEqual(sniff.total, 16)


class ResponseChunkTests(unittest.TestCase):

    def test_read1_is_preferred_without_waiting_for_a_full_buffer(self):
        response = upstream([])
        response.raw = mock.Mock(spec=["read1", "stream"])
        response.raw.read1.side_effect = [COMMENT, b""]
        self.assertEqual(list(devinx.response_chunks(response)), [COMMENT])
        response.raw.stream.assert_not_called()
        response.raw.read1.assert_has_calls([
            mock.call(65536, decode_content=True),
            mock.call(65536, decode_content=True)])

    def test_older_identity_reader_does_not_fill_a_decoded_buffer(self):
        response = upstream([COMMENT])
        self.assertEqual(list(devinx.response_chunks(response)), [COMMENT])
        response.raw.stream.assert_called_once_with(65536, decode_content=False)


class RelayStreamTests(RelayHarness, unittest.TestCase):
    """Messages SSE: a stream that dies inside a 200 is a stream_error."""

    def test_clean_stream_forwards_every_frame_and_reports_usage(self):
        usage = []
        request, log, status, headers, body = self.relay(
            [upstream(OK_STREAM)], on_usage=usage.append)
        self.assertTrue(status.startswith("HTTP/1.1 200"))
        self.assertEqual(body, b"".join(OK_STREAM))
        self.assertEqual(usage, [16])
        self.assertIn("status=200", log)

    def test_error_event_is_a_stream_error_not_a_200(self):
        request, log, status, headers, body = self.relay([upstream(
            [MESSAGE_START, CONTENT_START,
             error_event("overloaded_error", "slow down")])])
        self.assertIn("status=stream_error", log)
        self.assertNotIn("status=200", log)
        # The client gets the error event the real API would have sent.
        events = [f for f in body.split(b"\n\n") if f.strip()]
        self.assertEqual(events[-1].split(b"\n")[0], b"event: error")

    def test_eof_before_message_stop_is_a_stream_error(self):
        request, log, status, headers, body = self.relay(
            [upstream([MESSAGE_START, TEXT_DELTA])])
        self.assertIn("status=stream_error", log)
        self.assertIn("error_type=overloaded_error", log)
        self.assertIn("partial=True", log)
        self.assertIn(b'"type": "overloaded_error"', body)

    def test_no_data_message_stop_is_not_a_success(self):
        # Only a message_stop that carries its data ends the stream; a bare
        # "event: message_stop" line is forwarded and the run is still a
        # stream_error when the connection ends.
        request, log, status, headers, body = self.relay(
            [upstream([MESSAGE_START, TEXT_DELTA, b"event: message_stop\n\n"])])
        self.assertIn("status=stream_error", log)
        self.assertNotIn("status=200", log)
        self.assertIn(b"event: message_stop", body)
        self.assertIn(b"event: error", body)

    def test_event_and_data_type_mismatch_is_an_error(self):
        for frame in (b"event: message_stop\ndata: {\"type\":\"ping\"}\n\n",
                      b"event: ping\ndata: {\"type\":\"message_stop\"}\n\n",
                      b"data: {\"foo\":1}\n\n"):
            with self.subTest(frame=frame.split(b"\n")[0]):
                request, log, status, headers, body = self.relay(
                    [upstream([MESSAGE_START, TEXT_DELTA, frame])])
                self.assertIn("status=stream_error", log)
                self.assertIn("error_type=api_error", log)

    def test_malformed_error_shape_falls_back_safely(self):
        for error in ({"type": "error", "error": "oops"},
                      {"type": "error", "error": {"type": 42, "message": 7}},
                      {"type": "error",
                       "error": {"message": "no type at all"}}):
            with self.subTest(error=error):
                request, log, status, headers, body = self.relay(
                    [upstream([MESSAGE_START, devinx.sse_bytes(error)])])
                self.assertIn("status=stream_error", log)
                self.assertIn("error_type=api_error", log)
                self.assertIn(b"event: error", body)

    def test_error_log_keeps_metadata_but_never_the_message(self):
        request, log, status, headers, body = self.relay([upstream(
            [error_event("authentication_error", "sk-ant-secret-value",
                         code="bad_key")],
            headers={"request-id": "req-abc123"})])
        self.assertIn("error_type=authentication_error", log)
        self.assertIn("error_code=bad_key", log)
        self.assertIn("request_id=req-abc123", log)
        self.assertNotIn("sk-ant-secret-value", log)

    def test_stream_stops_reading_after_message_stop(self):
        seen = []

        def stream(*args, **kwargs):
            for chunk in OK_STREAM + [error_event("api_error"), b"extra"]:
                seen.append(chunk)
                yield chunk
        response = upstream(OK_STREAM)
        response.raw.stream.side_effect = stream
        request, log, status, headers, body = self.relay([response])
        self.assertEqual(seen, OK_STREAM, "read past message_stop")
        self.assertEqual(body, b"".join(OK_STREAM))
        self.assertIn("status=200", log)

    def test_malformed_json_frame_is_a_stream_error(self):
        request, log, status, headers, body = self.relay(
            [upstream([MESSAGE_START, b"data: {not json\n\n"])])
        self.assertIn("status=stream_error", log)
        self.assertIn("error_type=api_error", log)
        self.assertIn(b"event: error", body)

    def test_restarted_message_mid_stream_is_a_stream_error(self):
        request, log, status, headers, body = self.relay(
            [upstream([MESSAGE_START, MESSAGE_START])])
        self.assertIn("status=stream_error", log)
        self.assertIn("error_type=api_error", log)

    def test_oversized_frame_is_a_stream_error_not_a_crash(self):
        with mock.patch.object(devinx, "MAX_FRAME_BYTES", 64):
            request, log, status, headers, body = self.relay(
                [upstream([b"data: " + b"x" * 100 + b"\n\n"])])
        self.assertIn("status=stream_error", log)
        self.assertIn("transport=ValueError", log)
        self.assertIn(b"event: error", body)


class TransparencyTests(RelayHarness, unittest.TestCase):
    """What the stream fix touches must stay invisible everywhere else."""

    def test_stream_response_drops_encoding_and_length_headers(self):
        # Upstream says gzip+length anyway; what the client gets is the
        # decoded stream, so neither header may survive.
        request, log, status, headers, body = self.relay([upstream(
            OK_STREAM,
            headers={"content-encoding": "gzip", "content-length": "999",
                     "request-id": "req-1"})])
        self.assertNotIn("content-encoding", headers)
        self.assertNotIn("content-length", headers)
        self.assertEqual(headers.get("request-id"), "req-1")
        self.assertEqual(body, b"".join(OK_STREAM))
        sent = request.call_args.kwargs["headers"]
        self.assertEqual(sent["accept-encoding"], "identity")

    def test_non_sse_json_is_byte_for_byte(self):
        payload = b'{"id":"msg_1","type":"message"}'
        request, log, status, headers, body = self.relay([upstream(
            [payload], headers={"content-type": "application/json"})])
        self.assertTrue(status.startswith("HTTP/1.1 200"))
        self.assertEqual(headers.get("content-type"), "application/json")
        self.assertEqual(body, payload)
        self.assertIn("status=200", log)

    def test_http_error_status_and_body_are_relayed(self):
        payload = b'{"type":"error","error":{"type":"server_error"}}'
        request, log, status, headers, body = self.relay([upstream(
            [payload], status=500,
            headers={"content-type": "application/json"})])
        self.assertTrue(status.startswith("HTTP/1.1 500"))
        self.assertEqual(body, payload)
        self.assertIn("status=500", log)

    def test_codex_stream_is_not_inspected(self):
        # A native Codex stream is not Messages SSE: an error event and no
        # message_stop inside is still a clean 200 relay, verbatim.
        wire = [b"event: response.created\ndata: {}\n\n",
                error_event("api_error", "gone")]
        request, log, status, headers, body = self.relay(
            [upstream(wire)], label="codex")
        self.assertEqual(body, b"".join(wire))
        self.assertIn("status=200", log)
        self.assertNotIn("stream_error", log)


class GptRetryTests(RelayHarness, unittest.TestCase):
    """gpt-* only: one silent retry while no content has reached the client."""

    HEADERS = {"Authorization": "Bearer claude-secret",
               "X-Api-Key": "sk-ant-secret", "Cookie": "session=s",
               "X-Claude-Code-Session-Id": "sid-1"}

    def relay_gpt(self, responses):
        return self.relay(responses, label="gpt", headers=self.HEADERS,
                          strip=devinx.GPT_STRIPPED)

    def test_pre_content_error_retries_once(self):
        request, log, status, headers, body = self.relay_gpt([
            upstream([COMMENT, PING, MESSAGE_START,
                      error_event("api_error", "boom")]),
            upstream(OK_STREAM)])
        self.assertEqual(request.call_count, 2)
        self.assertIn("retry=1", log)
        self.assertIn("status=200", log)
        # The buffered message_start from attempt one was never sent, and the
        # retry's own is: exactly one on the wire, no duplicated headers.
        self.assertEqual(body.count(b"event: message_start"), 1)
        self.assertTrue(body.startswith(COMMENT + PING))
        self.assertTrue(body.endswith(b"".join(OK_STREAM)))
        self.assertTrue(status.startswith("HTTP/1.1 200"))

    def test_retry_sends_the_same_body_and_no_claude_credential(self):
        request, log, status, headers, body = self.relay_gpt([
            upstream([error_event("overloaded_error")]),
            upstream(OK_STREAM)])
        self.assertEqual(request.call_count, 2)
        first, second = request.call_args_list
        self.assertEqual(first.kwargs["data"], second.kwargs["data"])
        for call in (first, second):
            sent = {k.lower(): v for k, v in call.kwargs["headers"].items()}
            for secret in ("authorization", "x-api-key", "cookie"):
                self.assertNotIn(secret, sent)
            self.assertEqual(sent.get("x-claude-code-session-id"), "sid-1")

    def test_no_retry_once_content_is_exposed(self):
        for i, first_block in enumerate((
                [CONTENT_START, TEXT_DELTA],
                [THINKING_START],
                [TOOL_START])):
            with self.subTest(block=i):
                request, log, status, headers, body = self.relay_gpt([
                    upstream([MESSAGE_START] + first_block +
                             [error_event("api_error")]),
                    upstream(OK_STREAM)])
                self.assertEqual(request.call_count, 1)
                self.assertIn("retry=0", log)
                self.assertIn("status=stream_error", log)
                self.assertIn(b"event: error", body)

    def test_non_transient_error_is_not_retried(self):
        request, log, status, headers, body = self.relay_gpt([
            upstream([MESSAGE_START,
                      error_event("invalid_request_error", "bad request")]),
            upstream(OK_STREAM)])
        self.assertEqual(request.call_count, 1)
        self.assertIn("status=stream_error", log)

    def test_second_pre_content_failure_is_the_last(self):
        request, log, status, headers, body = self.relay_gpt([
            upstream([MESSAGE_START, error_event("api_error")]),
            upstream([MESSAGE_START, error_event("overloaded_error")]),
            upstream(OK_STREAM)])
        self.assertEqual(request.call_count, 2)
        self.assertIn("status=stream_error", log)
        self.assertIn(b"event: error", body)
        self.assertNotIn(b"event: message_start", body)

    def test_second_attempt_http_error_becomes_an_sse_error(self):
        request, log, status, headers, body = self.relay_gpt([
            upstream([MESSAGE_START, error_event("api_error")]),
            upstream([b'{"type":"error","error":{"type":"server_error",'
                      b'"message":"down"}}'],
                     status=500, headers={"content-type": "application/json"})])
        self.assertEqual(request.call_count, 2)
        self.assertIn(b"event: error", body)
        self.assertIn(b"server_error", body)

    def test_committed_retry_never_sends_a_second_status_line(self):
        # A context refusal on the already-committed retry must become an SSE
        # error event, not another HTTP response down the same stream.
        def on_error(status, text):
            return devinx.RETRY
        request, log, status, headers, body = self.relay(
            [upstream([MESSAGE_START, error_event("api_error")]),
             upstream([b'{"type":"error"}'], status=413,
                      headers={"content-type": "application/json"})],
            label="gpt", headers=self.HEADERS, strip=devinx.GPT_STRIPPED,
            on_error=on_error)
        self.assertEqual(request.call_count, 2)
        self.assertNotIn(b"HTTP/1.1", body)
        self.assertIn(b"event: error", body)
        self.assertIn(b"api_error", body)

    def test_committed_retry_with_invalid_raw_body_still_sends_an_sse_error(self):
        for raw in (b"not-json", b"[]"):
            with self.subTest(raw=raw):
                request, log, status, headers, body = self.relay(
                    [upstream([MESSAGE_START, error_event("api_error")]),
                     upstream([b"context exceeded"], status=413,
                              headers={"content-type": "application/json"})],
                    label="gpt", raw=raw, on_error=lambda status, text: devinx.RETRY)
                self.assertEqual(request.call_count, 2)
                self.assertNotIn(b"HTTP/1.1", body)
                self.assertIn(b"event: error", body)
                self.assertIn(b"Upstream retry failed", body)

    def test_transport_error_before_content_retries(self):
        request, log, status, headers, body = self.relay_gpt([
            upstream([MESSAGE_START],
                     exc=urllib3.exceptions.ProtocolError("broken")),
            upstream(OK_STREAM)])
        self.assertEqual(request.call_count, 2)
        self.assertIn("transport=ProtocolError", log)
        self.assertEqual(body.count(b"event: message_start"), 1)

    def test_transport_error_after_content_does_not_retry(self):
        request, log, status, headers, body = self.relay_gpt([
            upstream([MESSAGE_START, CONTENT_START, TEXT_DELTA],
                     exc=urllib3.exceptions.ProtocolError("broken")),
            upstream(OK_STREAM)])
        self.assertEqual(request.call_count, 1)
        self.assertIn("status=stream_error", log)
        self.assertIn(b"event: error", body)

    def test_eof_before_content_retries(self):
        request, log, status, headers, body = self.relay_gpt([
            upstream([MESSAGE_START]),
            upstream(OK_STREAM)])
        self.assertEqual(request.call_count, 2)
        self.assertEqual(body.count(b"event: message_start"), 1)


class ScriptedUpstream:
    """One canned raw reply per request, on an ephemeral loopback port."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        outer = self

        class Http(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                outer.seen.append({
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": self.rfile.read(length)})
                reply = outer.replies[min(len(outer.seen),
                                        len(outer.replies)) - 1]
                self.send_response(reply["status"])
                for name, value in reply.get("headers", {}).items():
                    self.send_header(name, value)
                self.send_header("content-length", str(len(reply["body"])))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(reply["body"])

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Http)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def post(port, path, body, headers=()):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("POST", path, body=json.dumps(body).encode(),
                     headers={"content-type": "application/json",
                              **dict(headers)})
        resp = conn.getresponse()
        return (resp.status,
                {k.lower(): v for k, v in resp.getheaders()},
                resp.read())
    finally:
        conn.close()


class LiveStreamTests(unittest.TestCase):
    """The whole path: client socket -> Handler -> real HTTP upstream."""

    def setUp(self):
        self.live = LiveServer()
        self.addCleanup(self.live.close)

    def test_upstream_gzip_stream_is_decoded_and_unlabelled(self):
        # The upstream ignored accept-encoding: identity and gzipped anyway.
        # The client must get plain SSE with neither the encoding nor the
        # (now meaningless) length forwarded.
        upstream_server = ScriptedUpstream([{
            "status": 200,
            "headers": {"content-type": "text/event-stream",
                        "content-encoding": "gzip"},
            "body": gzip.compress(b"".join(OK_STREAM))}])
        self.addCleanup(upstream_server.close)
        with mock.patch.object(devinx, "CLAUDE_UPSTREAM", upstream_server.url):
            status, headers, body = post(
                self.live.port, "/v1/messages",
                {"model": "claude-test", "max_tokens": 8,
                 "messages": [{"role": "user", "content": "hi"}]},
                {"authorization": "Bearer test"})
        self.assertEqual(status, 200)
        self.assertNotIn("content-encoding", headers)
        self.assertNotIn("content-length", headers)
        self.assertEqual(body, b"".join(OK_STREAM))
        sent = upstream_server.seen[0]["headers"]
        self.assertEqual(sent.get("accept-encoding"), "identity")

    def test_non_chunked_stream_forwards_heartbeats_before_eof(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                release = threading.Event()
                prefix = COMMENT + MESSAGE_START + CONTENT_START + TEXT_DELTA
                suffix = BLOCK_STOP + MESSAGE_DELTA + MESSAGE_STOP
                if compressed:
                    compressor = zlib.compressobj(wbits=31)
                    first = compressor.compress(prefix) + compressor.flush(zlib.Z_SYNC_FLUSH)
                    last = compressor.compress(suffix) + compressor.flush(zlib.Z_FINISH)
                else:
                    first, last = prefix, suffix

                class Slow(http.server.BaseHTTPRequestHandler):
                    protocol_version = "HTTP/1.1"

                    def do_POST(self):
                        self.rfile.read(int(self.headers.get("content-length") or 0))
                        self.send_response(200)
                        self.send_header("content-type", "text/event-stream")
                        self.send_header("content-length", str(len(first) + len(last)))
                        if compressed:
                            self.send_header("content-encoding", "gzip")
                        self.send_header("connection", "close")
                        self.end_headers()
                        self.wfile.write(first)
                        self.wfile.flush()
                        release.wait(5)
                        self.wfile.write(last)
                        self.wfile.flush()

                    def log_message(self, *args):
                        pass

                upstream_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
                threading.Thread(target=upstream_server.serve_forever, daemon=True).start()
                url = f"http://127.0.0.1:{upstream_server.server_address[1]}"
                conn = http.client.HTTPConnection("127.0.0.1", self.live.port, timeout=2)
                try:
                    with mock.patch.object(devinx, "CLAUDE_UPSTREAM", url):
                        conn.request("POST", "/v1/messages", body=json.dumps({
                            "model": "claude-test", "stream": True,
                            "messages": [{"role": "user", "content": "hi"}]}),
                            headers={"content-type": "application/json",
                                     "authorization": "Bearer test"})
                        response = conn.getresponse()
                        self.assertEqual(response.readline(), b": keep-alive\n")
                        self.assertEqual(response.readline(), b"\n")
                        release.set()
                        self.assertEqual(response.read(), prefix[len(COMMENT):] + suffix)
                finally:
                    release.set()
                    conn.close()
                    upstream_server.shutdown()
                    upstream_server.server_close()

    def test_gpt_route_retries_a_pre_content_stream_error(self):
        upstream_server = ScriptedUpstream([
            {"status": 200,
             "headers": {"content-type": "text/event-stream"},
             "body": MESSAGE_START + error_event("api_error", "boom")},
            {"status": 200,
             "headers": {"content-type": "text/event-stream"},
             "body": b"".join(OK_STREAM)}])
        self.addCleanup(upstream_server.close)
        with mock.patch.object(devinx, "GPT_UPSTREAM", upstream_server.url):
            status, headers, body = post(
                self.live.port, "/v1/messages",
                {"model": "gpt-6-luna", "max_tokens": 8, "stream": True,
                 "messages": [{"role": "user", "content": "hi"}]},
                {"authorization": "Bearer claude-secret",
                 "x-api-key": "sk-ant-secret"})
        self.assertEqual(status, 200)
        self.assertEqual(len(upstream_server.seen), 2)
        self.assertEqual(upstream_server.seen[0]["body"],
                         upstream_server.seen[1]["body"])
        for seen in upstream_server.seen:
            for secret in ("authorization", "x-api-key", "cookie"):
                self.assertNotIn(secret, seen["headers"])
        self.assertEqual(body.count(b"event: message_start"), 1)
        self.assertIn(b"event: message_stop", body)


if __name__ == "__main__":
    unittest.main()
