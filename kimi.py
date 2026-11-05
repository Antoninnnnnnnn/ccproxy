"""Kimi Code accounts and the kimi-* route's request and response shaping.

A Kimi Code subscription comes with an API key (`sk-kimi-...`) for its coding
endpoint, which speaks the Anthropic Messages API natively: thinking with
signatures, tool use, prompt caching, count_tokens. So unlike gpt-* nothing is
translated; devinx relays the turn with the subscription's key in place of the
client's own credential, the way it does SWE-2 with a Devin login.

The keys live in `<data dir>/kimi/<name>.key`, one account per file, written by
`devinx --kimi-add`. `DEVINX_KIMI_KEYS` (comma-separated) replaces the files.
Standard library only: the launcher imports this to decide whether to offer the
Kimi rows and agents at all, and the add command validates a key with urllib.
"""
import getpass
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request

UPSTREAM = os.environ.get("DEVINX_KIMI_UPSTREAM", "https://api.kimi.com/coding")
PREFIX = "kimi-"
# Kimi's thinking signatures are opaque base64 with nothing to tell them from
# Anthropic's, which refuses any it did not sign. Every signature that leaves
# the kimi-* route is tagged, so a conversation moved to claude-* can drop it,
# and the tag is taken off again before the history goes back to Kimi.
SIGNATURE_PREFIX = "kimi:"

# (upstream id, label, window). The upstream answers 200 for any model name
# at all, falling back to its default, so a typo would silently run another
# model: only ids listed here, or in the live catalog, are relayed.
MODELS = (
    ("k3.1-pro", "K3.1 Pro", 1048576),
    ("k3.1-pro-highspeed", "K3.1 Pro Highspeed", 1048576),
    ("k3", "K3", 1048576),
    ("k3-256k", "K3 256k", 262144),
    ("kimi-for-coding", "K2.8 Preview", 1048576),
    ("kimi-for-coding-highspeed", "K2.7 Code Highspeed", 262144),
)
# Kimi reasons at low, high and max. Claude Code's slider (lent by Opus) also
# has medium and xhigh; each goes to the level above it.
EFFORTS = {"low": "low", "medium": "high", "high": "high",
           "xhigh": "max", "max": "max"}
COMPACT_AT = (int(os.environ["DEVINX_KIMI_COMPACT_AT"])
              if os.environ.get("DEVINX_KIMI_COMPACT_AT") else None)
KEY_SHAPE = re.compile(r"^sk-kimi-[A-Za-z0-9]{16,}$")
NAME_SHAPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

_live = {"at": 0.0, "models": {}}


def client_id(upstream):
    """The id Claude Code uses: `kimi-` + the upstream id, once."""
    return upstream if upstream.startswith(PREFIX) else PREFIX + upstream


def known_models():
    """{client id: (upstream id, label, window)}, the static table plus
    whatever the live catalog has added."""
    out = {client_id(up): (up, label, window) for up, label, window in MODELS}
    for up, (label, window) in _live["models"].items():
        out.setdefault(client_id(up), (up, label, window))
    return out


def upstream_model(model):
    """The upstream id for a kimi-* client id, or None when it is unknown."""
    hit = known_models().get(model)
    return hit[0] if hit else None


def context_tokens(model):
    hit = known_models().get(model)
    return hit[2] if hit else 262144


def compact_at(model):
    return COMPACT_AT if COMPACT_AT is not None else context_tokens(model) * 9 // 10


def refresh_catalog(key, session=None, every=600.0):
    """Learn models the static table does not list yet, at most every
    `every` seconds. Failure keeps what is known."""
    if not key or time.time() - _live["at"] < every:
        return False
    _live["at"] = time.time()
    try:
        if session is not None:
            r = session.get(UPSTREAM + "/v1/models", timeout=(5, 15),
                            headers={"authorization": f"Bearer {key}"})
            data = r.json() if r.status_code == 200 else {}
        else:
            data = _get_json("/v1/models", key)
        models = {}
        for m in data.get("data") or []:
            if isinstance(m, dict) and isinstance(m.get("id"), str):
                window = m.get("context_length") or 262144
                models[m["id"]] = (str(m.get("display_name") or m["id"]),
                                   int(window))
        if models:
            _live["models"] = models
            return True
    except Exception:
        pass
    return False


# --- request and response shaping ------------------------------------------

def _unwrap(block):
    sig = block.get("signature")
    if (isinstance(block, dict) and block.get("type") == "thinking"
            and isinstance(sig, str) and sig.startswith(SIGNATURE_PREFIX)):
        return {**block, "signature": sig[len(SIGNATURE_PREFIX):]}
    return block


def prepare(body, upstream):
    """The body as Kimi takes it. The same object when nothing changes.

    - the model is the upstream id;
    - a thinking block Kimi signed gets its own signature back;
    - effort is one of Kimi's three levels;
    - `thinking: disabled` is dropped: K3.1 Pro refuses it ("does not
      support disabled thinking mode") and every current model thinks always;
    - `tool_addition` blocks are dropped: Claude Code announces a tool loaded
      mid-conversation with one, inside a `{role: "system"}` turn, and Kimi
      refuses the whole request over it ("Invalid request Error") while it
      takes the system turn's text. The tool itself is already in `tools`.
    """
    out = dict(body)
    out["model"] = upstream
    thinking = out.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        out.pop("thinking")
    config = out.get("output_config")
    if isinstance(config, dict) and config.get("effort") in EFFORTS:
        level = EFFORTS[config["effort"]]
        if level != config["effort"]:
            out["output_config"] = {**config, "effort": level}
    messages = out.get("messages")
    if isinstance(messages, list):
        fixed = []
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, list):
                blocks = [_unwrap(b) if isinstance(b, dict) else b for b in content
                          if not (isinstance(b, dict)
                                  and b.get("type") == "tool_addition")]
                if not blocks:
                    continue
                if len(blocks) != len(content) or any(
                        a is not b for a, b in zip(blocks, content)):
                    m = {**m, "content": blocks}
            fixed.append(m)
        out["messages"] = fixed
    return out


class SignatureTagger:
    """Tags the signature of each thinking block in a Messages SSE stream.

    A signature can arrive in content_block_start or in one or more
    signature_delta events; the client concatenates them, so only the first
    non-empty piece of each block is tagged."""

    def __init__(self):
        self.tagged = set()

    def event(self, data):
        """The event with its signature tagged, or None when it carries none."""
        if not isinstance(data, dict):
            return None
        kind = data.get("type")
        index = data.get("index")
        if kind == "content_block_start":
            block = data.get("content_block")
            if (isinstance(block, dict) and block.get("type") == "thinking"
                    and isinstance(block.get("signature"), str)
                    and block["signature"] and index not in self.tagged):
                self.tagged.add(index)
                return {**data, "content_block": {
                    **block, "signature": SIGNATURE_PREFIX + block["signature"]}}
            return None
        if kind == "content_block_delta":
            delta = data.get("delta")
            if (isinstance(delta, dict) and delta.get("type") == "signature_delta"
                    and isinstance(delta.get("signature"), str)
                    and delta["signature"] and index not in self.tagged):
                self.tagged.add(index)
                return {**data, "delta": {
                    **delta, "signature": SIGNATURE_PREFIX + delta["signature"]}}
        return None


def tag_message(message):
    """A non-streamed response with its thinking signatures tagged."""
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return message
    blocks = []
    for b in content:
        if (isinstance(b, dict) and b.get("type") == "thinking"
                and isinstance(b.get("signature"), str) and b["signature"]
                and not b["signature"].startswith(SIGNATURE_PREFIX)):
            b = {**b, "signature": SIGNATURE_PREFIX + b["signature"]}
        blocks.append(b)
    return {**message, "content": blocks}


def overflow(status, text):
    """The window figure from Kimi's context refusal, or None for any other
    error. Measured: 400 invalid_request_error, "Your request exceeded
    k3-256k model token limit: 262144"."""
    if status not in (400, 413):
        return None
    match = re.search(r"exceeded .{0,80}?token limit[^0-9]{0,8}(\d+)", text)
    return int(match.group(1)) if match else None


_QUOTA_WORDS = ("quota", "rate limit", "rate_limit", "too many requests",
                "usage limit", "exceeded your", "insufficient balance",
                "billing", "concurrency")


def account_refusal(status, text):
    """'auth' when the key itself is refused, 'quota' when the account is
    spent or throttled, None for an error about the request."""
    lowered = text.lower()
    if status in (401,) or (status == 403 and ("auth" in lowered
                                                or "key" in lowered)):
        return "auth"
    if status == 429 or (status in (402, 403) and any(
            w in lowered for w in _QUOTA_WORDS)):
        return "quota"
    if status == 400 and any(w in lowered for w in _QUOTA_WORDS) \
            and overflow(status, text) is None:
        return "quota"
    return None


# --- accounts ----------------------------------------------------------------

def keys_dir(data_dir):
    return os.path.join(data_dir, "kimi")


def _key_files(data_dir):
    try:
        names = sorted(os.listdir(keys_dir(data_dir)))
    except OSError:
        return []
    return [os.path.join(keys_dir(data_dir), n) for n in names
            if n.endswith(".key") and not n.startswith(".")]


def load_accounts(data_dir):
    """[{'name', 'key', 'path'}], from DEVINX_KIMI_KEYS or the key files."""
    env = os.environ.get("DEVINX_KIMI_KEYS", "")
    if env.strip():
        keys = [k.strip() for k in env.split(",") if k.strip()]
        return [{"name": f"kimi-env{i + 1}", "key": k, "path": None}
                for i, k in enumerate(keys)]
    out = []
    for path in _key_files(data_dir):
        try:
            with open(path, encoding="utf-8") as fh:
                key = fh.read().strip()
        except OSError:
            continue
        if key:
            out.append({"name": os.path.basename(path)[:-len(".key")],
                        "key": key, "path": path})
    return out


def signature(data_dir):
    """What changes when an account is added, removed or rewritten."""
    if os.environ.get("DEVINX_KIMI_KEYS", "").strip():
        return ("env", os.environ["DEVINX_KIMI_KEYS"])
    out = []
    for path in _key_files(data_dir):
        try:
            out.append((path, os.path.getmtime(path)))
        except OSError:
            pass
    return tuple(out)


def configured(data_dir):
    return bool(load_accounts(data_dir))


class Pool:
    """The Kimi accounts the service rotates through.

    Turns go round robin, least recently used first. An account refused for
    quota is blocked until its window resets (read from /v1/usages when the
    refusal does not say); a key the upstream rejects is set aside until its
    file is rewritten. Accounts keep that state across a rescan."""

    def __init__(self, data_dir, rescan_every=30.0):
        import threading
        self.data_dir = data_dir
        self.rescan_every = rescan_every
        self.lock = threading.Lock()
        self.accounts = []
        self.scanned_at = 0.0
        self.sig = None

    def _rescan(self, now):
        if self.accounts and now - self.scanned_at < self.rescan_every:
            return
        self.scanned_at = now
        sig = signature(self.data_dir)
        if sig == self.sig and self.accounts:
            return
        fresh = load_accounts(self.data_dir)
        self.sig = sig
        known = {a["key"]: a for a in self.accounts}
        merged = []
        for f in fresh:
            a = known.get(f["key"])
            if a is None:
                a = {**f, "blocked_until": 0.0, "used_at": 0.0, "invalid": None,
                     "usage": None, "usage_at": 0.0}
                print(f"devinx: Kimi account {f['name']} loaded", flush=True)
            else:
                a["name"], a["path"] = f["name"], f["path"]
                if a.get("invalid") is not None and _mtime(f["path"]) != a["invalid"]:
                    a["invalid"] = None
            merged.append(a)
        for a in self.accounts:
            if a["key"] not in {f["key"] for f in fresh}:
                print(f"devinx: Kimi account {a['name']} removed", flush=True)
        self.accounts = merged

    def snapshot(self):
        with self.lock:
            self._rescan(time.time())
            return list(self.accounts)

    def claim(self, avoid=()):
        """(account, None), or (None, seconds until one is free) — None
        seconds when there is no usable account at all."""
        now = time.time()
        with self.lock:
            self._rescan(now)
            usable = [a for a in self.accounts if a["invalid"] is None]
            free = [a for a in usable
                    if a["blocked_until"] <= now and a["key"] not in avoid]
            if free:
                pick = min(free, key=lambda a: a["used_at"])
                pick["used_at"] = now
                return pick, None
            if usable:
                return None, max(1.0, min(a["blocked_until"] for a in usable) - now)
            return None, None

    def block(self, acct, seconds, why=""):
        with self.lock:
            acct["blocked_until"] = max(acct["blocked_until"], time.time() + seconds)
        print(f"devinx: Kimi account {acct['name']} blocked {seconds:.0f}s"
              f"{' (' + why + ')' if why else ''}", flush=True)

    def invalidate(self, acct):
        with self.lock:
            acct["invalid"] = _mtime(acct.get("path"))
        print(f"devinx: Kimi account {acct['name']} refused by the upstream; "
              f"set aside until its key is replaced (devinx --kimi-add "
              f"{acct['name']})", flush=True)


def _mtime(path):
    try:
        return os.path.getmtime(path) if path else 0.0
    except OSError:
        return 0.0


def _get_json(path, key, timeout=15):
    req = urllib.request.Request(UPSTREAM + path, headers={
        "authorization": f"Bearer {key}", "user-agent": "devinx"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _parse_time(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r"\.\d+", "", text).replace("Z", "+00:00")
    try:
        from datetime import datetime
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def parse_usage(data):
    """{'5h': (used ratio, reset epoch), 'month': (...)} from /v1/usages."""
    usages = (data or {}).get("usages") or {}
    out = {}
    for name, field in (("5h", "limit_5h"), ("month", "limit_month_total")):
        item = usages.get(field)
        if isinstance(item, dict):
            try:
                ratio = float(item.get("used_ratio") or 0.0)
            except (TypeError, ValueError):
                ratio = 0.0
            out[name] = (ratio, _parse_time(item.get("reset_time")))
    return out


def usage(acct, session=None, every=60.0):
    """The account's usage, fetched at most every `every` seconds."""
    now = time.time()
    if acct.get("usage") is not None and now - acct.get("usage_at", 0) < every:
        return acct["usage"]
    acct["usage_at"] = now
    try:
        if session is not None:
            r = session.get(UPSTREAM + "/v1/usages", timeout=(5, 10),
                            headers={"authorization": f"Bearer {acct['key']}"})
            data = r.json() if r.status_code == 200 else None
        else:
            data = _get_json("/v1/usages", acct["key"], timeout=10)
        if data is not None:
            acct["usage"] = parse_usage(data)
    except Exception:
        pass
    return acct.get("usage")


def block_seconds(acct, retry_after=None, session=None):
    """How long a quota refusal should keep an account out of rotation: what
    the refusal said, else until the spent window resets, else 60s."""
    if retry_after:
        try:
            return max(5.0, float(retry_after))
        except ValueError:
            pass
    acct["usage_at"] = 0.0
    info = usage(acct, session=session) or {}
    now = time.time()
    for name in ("5h", "month"):
        ratio, reset = info.get(name, (0.0, None))
        if ratio >= 0.999 and reset and reset > now:
            return reset - now + 5
    return 60.0


# --- the add / list / remove commands -----------------------------------------

def _mask(key):
    return key[:8] + "…" + key[-4:] if len(key) > 16 else "…"


def _describe(info):
    parts = []
    for name in ("5h", "month"):
        if name in info:
            ratio, reset = info[name]
            when = (time.strftime(" (reset %d/%m %H:%M)", time.localtime(reset))
                    if reset else "")
            parts.append(f"{name} {ratio * 100:.1f}%{when}")
    return ", ".join(parts) or "usage unknown"


def add(data_dir, name=None, key=None, out=sys.stdout):
    if key is None:
        key = (getpass.getpass("Kimi Code API key (sk-kimi-...): ")
               if sys.stdin.isatty() else sys.stdin.readline())
    key = key.strip()
    if not KEY_SHAPE.match(key):
        print("devinx: that does not look like a Kimi Code key (sk-kimi-...)",
              file=sys.stderr)
        return 2
    try:
        data = _get_json("/v1/usages", key)
    except urllib.error.HTTPError as e:
        print(f"devinx: the key was refused by Kimi ({e.code}); nothing saved",
              file=sys.stderr)
        return 1
    except Exception as e:
        print(f"devinx: could not reach Kimi to check the key "
              f"({type(e).__name__}); nothing saved", file=sys.stderr)
        return 1
    existing = load_accounts(data_dir) if not os.environ.get("DEVINX_KIMI_KEYS") else []
    if name is None:
        same = [a for a in existing if a["key"] == key]
        name = same[0]["name"] if same else f"kimi{len(existing) + 1}"
        taken = {a["name"] for a in existing} - {name}
        n = len(existing) + 1
        while name in taken:
            n += 1
            name = f"kimi{n}"
    if not NAME_SHAPE.match(name):
        print("devinx: an account name is letters, digits, '.', '_' or '-'",
              file=sys.stderr)
        return 2
    folder = keys_dir(data_dir)
    os.makedirs(folder, mode=0o700, exist_ok=True)
    for other in existing:
        if other["key"] == key and other["name"] != name and other["path"]:
            os.remove(other["path"])
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-")
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(key + "\n")
        os.replace(tmp, os.path.join(folder, name + ".key"))
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    print(f"Kimi account {name} saved ({_mask(key)}): "
          f"{_describe(parse_usage(data))}", file=out)
    if os.environ.get("DEVINX_KIMI_KEYS"):
        print("note: DEVINX_KIMI_KEYS is set and replaces the saved keys",
              file=out)
    return 0


def list_accounts(data_dir, out=sys.stdout):
    accounts = load_accounts(data_dir)
    if not accounts:
        print("no Kimi account: add one with `devinx --kimi-add`", file=out)
        return 1
    for a in accounts:
        try:
            info = _describe(parse_usage(_get_json("/v1/usages", a["key"])))
        except urllib.error.HTTPError as e:
            info = f"refused ({e.code})"
        except Exception as e:
            info = f"unreachable ({type(e).__name__})"
        print(f"{a['name']:<16} {_mask(a['key'])}  {info}", file=out)
    return 0


def remove(data_dir, name, out=sys.stdout):
    path = os.path.join(keys_dir(data_dir), f"{name}.key")
    if not NAME_SHAPE.match(name or "") or not os.path.exists(path):
        print(f"devinx: no Kimi account named {name!r}", file=sys.stderr)
        return 1
    os.remove(path)
    print(f"Kimi account {name} removed", file=out)
    return 0


def main(flag, rest, data_dir):
    """`devinx --kimi-add [name]`, `--kimi-list`, `--kimi-remove name`."""
    if flag == "--kimi-add":
        return add(data_dir, rest[0] if rest else None)
    if flag == "--kimi-list":
        return list_accounts(data_dir)
    if flag == "--kimi-remove":
        if not rest:
            print("usage: devinx --kimi-remove <name>", file=sys.stderr)
            return 2
        return remove(data_dir, rest[0])
    return 2
