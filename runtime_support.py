"""Small standard-library helpers shared by service, launcher and diagnostics."""
import contextlib
import errno
import hashlib
import os
from pathlib import Path
import sys
import time

# The project was devinx before it served several providers. Everything that
# name left on a machine keeps working: DEVINX_* variables are read as their
# CCPROXY_* counterparts, and the data directory moves on first launch.
NAME = 'ccproxy'
LEGACY_NAME = 'devinx'
LEGACY_ENV, ENV_PREFIX = 'DEVINX_', 'CCPROXY_'


def mirror_legacy_env(environ=None):
    """Copy each DEVINX_* variable to its CCPROXY_* name, unless that one is
    set already (the new name wins). Runs at import: service, launcher,
    installer and diagnostics all import this module first, and the service's
    children inherit the result."""
    environ = os.environ if environ is None else environ
    for key in list(environ):
        if key.startswith(LEGACY_ENV):
            environ.setdefault(ENV_PREFIX + key[len(LEGACY_ENV):], environ[key])


mirror_legacy_env()


def _data_base():
    if sys.platform == 'win32':
        return os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    if sys.platform == 'darwin':
        return os.path.expanduser('~/Library/Application Support')
    return os.environ.get('XDG_DATA_HOME') or os.path.expanduser('~/.local/share')


_LEGACY_MARKERS = ('service-secret', 'devinx.log', 'dashboard-token', 'kimi')


def _legacy_data(path):
    """Whether `path` is a devinx data directory, and only that.

    Where LOCALAPPDATA is missing the Windows base falls back to the home
    directory, and `~/devinx` there is as likely to be the repository itself:
    measured, a run that took it for the data directory moved the checkout.
    So a directory is taken for data only when it holds what the service
    writes, and never when it is a git checkout."""
    if not os.path.isdir(path) or os.path.islink(path):
        return False
    if os.path.exists(os.path.join(path, '.git')):
        return False
    return any(os.path.exists(os.path.join(path, m)) for m in _LEGACY_MARKERS)


def data_dir():
    """Per-OS writable location for credentials, logs and runtime state.

    The old devinx directory is used as long as it has not been moved (see
    migrate_data_dir), so nothing is lost if the move could not happen."""
    if os.environ.get('CCPROXY_DATA'):
        return os.path.normpath(os.environ['CCPROXY_DATA'])
    base = _data_base()
    new = os.path.normpath(os.path.join(base, NAME))
    old = os.path.normpath(os.path.join(base, LEGACY_NAME))
    if not os.path.exists(new) and _legacy_data(old):
        return old
    return new


def migrate_data_dir():
    """Move the devinx data directory, and its log, to the ccproxy names.

    Called by the launcher before it starts anything. Where the platform
    allows it a link is left at each old name, so a devinx service still
    running from before the rename keeps its credentials, log and lock until
    it is replaced. A move that fails (Windows, with that service holding its
    files) changes nothing: data_dir() keeps answering the old directory."""
    if os.environ.get('CCPROXY_DATA'):
        return data_dir()
    base = _data_base()
    new = os.path.join(base, NAME)
    old = os.path.join(base, LEGACY_NAME)
    if not os.path.exists(new) and _legacy_data(old):
        try:
            os.rename(old, new)
        except OSError:
            return data_dir()
        if os.name == 'posix':
            with contextlib.suppress(OSError):
                os.symlink(NAME, old)
    here = data_dir()
    log, old_log = os.path.join(here, NAME + '.log'), os.path.join(here, LEGACY_NAME + '.log')
    if not os.path.exists(log) and os.path.isfile(old_log) and not os.path.islink(old_log):
        try:
            os.rename(old_log, log)
        except OSError:
            return here
        if os.name == 'posix':
            with contextlib.suppress(OSError):
                os.symlink(NAME + '.log', old_log)
    return here

# Runtime values that can be disclosed safely. Never add credentials, headers,
# prompts, account identifiers or arbitrary environment variables here.
CONFIG_FIELDS = {
    'CCPROXY_CONTEXT_TOKENS': 'SWE_CONTEXT_TOKENS',
    'CCPROXY_COMPACT_AT': 'COMPACT_AT',
    'CCPROXY_COMPACT_STRICT': 'COMPACT_STRICT',
    'CCPROXY_MAX_BODY': 'MAX_BODY_BYTES',
    'CCPROXY_MAX_FRAME': 'MAX_FRAME_BYTES',
    'CCPROXY_MAX_INFLATED_FRAME': 'MAX_INFLATED_FRAME_BYTES',
    'CCPROXY_MAX_INFLIGHT': 'MAX_INFLIGHT',
    'CCPROXY_MAX_SWE_INFLIGHT': 'MAX_SWE_INFLIGHT',
    'CCPROXY_HTTP_READ_TIMEOUT': 'HTTP_READ_TIMEOUT',
    'CCPROXY_CLIENT_WRITE_TIMEOUT': 'CLIENT_WRITE_TIMEOUT',
    'CCPROXY_RELAY_READ_TIMEOUT': 'RELAY_READ_TIMEOUT',
    'CCPROXY_KEEPALIVE': 'KEEPALIVE_EVERY',
    'CCPROXY_RATE_WAIT': 'RATE_WAIT_BUDGET',
    'CCPROXY_RATE_FLOOR': 'RATE_FLOOR',
    'CCPROXY_DRAIN': 'DRAIN_SECONDS',
}


def swe_enabled():
    """Whether Devin / SWE-2 access is switched on.

    Off unless CCPROXY_SWE=1, in the service and in the launcher alike: they
    read this one function, so a session never advertises agents the service
    would refuse. Read at call time, not import time, so a test can flip it.
    """
    return os.environ.get('CCPROXY_SWE') == '1'


# The libraries the service's behaviour depends on. A dependency upgrade is a
# new service as much as a code change is, and must read as one.
DEPENDENCIES = ('protobuf', 'requests', 'urllib3')


def _dependency_versions():
    try:
        from importlib import metadata
    except ImportError:
        return []
    out = []
    for name in DEPENDENCIES:
        try:
            out.append(f'{name}={metadata.version(name)}')
        except Exception:
            out.append(f'{name}=absent')
    return out


def build_id(directory):
    """Fingerprint shipped runtime code/descriptors, not local state or secrets.

    A file that cannot be read is fingerprinted by its absence rather than
    turning the whole id into 'unknown': 'unknown' switches stale-service
    detection off altogether, so one missing dashboard file used to leave
    last week's service serving today's launcher without a word. Launcher and
    service both call this, from the same interpreter, so they agree.
    """
    root = Path(directory)
    files = [root / name for name in (
        'ccproxy.py', 'runtime_support.py', 'diagnostics.py', 'kimi.py',
        'tools/log_stats.py', 'tools/dashboard.html')]
    try:
        files.extend(sorted((root / 'descriptors').glob('*.fdp')))
    except OSError:
        pass
    digest = hashlib.sha256()
    for path in files:
        name = str(path.relative_to(root)).replace('\\', '/').encode()
        try:
            data = path.read_bytes()
        except OSError:
            digest.update(b'missing\0' + name + b'\0')
            continue
        digest.update(name)
        digest.update(b'\0')
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)
    for version in _dependency_versions():
        digest.update(b'dep\0' + version.encode() + b'\0')
    return digest.hexdigest()[:12]


@contextlib.contextmanager
def startup_lock(directory, port, timeout=40):
    """Serialize launchers sharing a data directory and endpoint.

    Do not unlink the lock on release: a waiting launcher holds that same inode.
    OS locks release automatically if the holder exits. This is coordination,
    not a security boundary between users of the machine.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'launch-{port}.lock')
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    acquired = False
    try:
        if os.name == 'nt':
            import msvcrt
            if os.fstat(fd).st_size == 0:
                os.write(fd, b'0')
            def lock():
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            def unlock():
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            def lock():
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def unlock():
                fcntl.flock(fd, fcntl.LOCK_UN)
        deadline = time.monotonic() + timeout
        while True:
            try:
                lock()
                acquired = True
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError('Another launcher is still starting ccproxy') from error
                time.sleep(0.05)
        yield
    finally:
        try:
            if acquired:
                unlock()
        finally:
            os.close(fd)


def _lock_dir():
    """A per-user directory whose location does not depend on the shell.

    The port lock has to be the same file for every ccproxy of this user on
    this machine, whatever CCPROXY_DATA, XDG_DATA_HOME or TMPDIR a given shell
    set, or two services with different settings both believe they own the
    port. /run/user/<uid> exists on systemd hosts and is private by
    construction; elsewhere a directory under the home is.
    """
    uid = getattr(os, 'getuid', lambda: None)()
    run = f'/run/user/{uid}' if uid is not None else None
    if run and os.path.isdir(run) and os.access(run, os.W_OK):
        base = os.path.join(run, 'ccproxy')
    elif os.name == 'nt':
        base = os.path.join(os.environ.get('LOCALAPPDATA') or os.path.expanduser('~'),
                            'ccproxy', 'run')
    else:
        base = os.path.join(os.path.expanduser('~'), '.cache', 'ccproxy', 'run')
    os.makedirs(base, mode=0o700, exist_ok=True)
    return base


class PortLock:
    """Held by the service for as long as it listens on its port.

    SO_REUSEPORT used to let a second ccproxy bind the same port — started
    with another data directory, by hand, or by a launcher whose probe timed
    out under load — after which the kernel spread connections between two
    processes, possibly two builds, each with its own rate-limit state. The
    lock is what makes "one service per port" true. It is released the
    moment the service stops listening, not when it exits, so a successor
    can start while the old process is still draining its turns.
    """

    def __init__(self, port):
        self.path = os.path.join(_lock_dir(), f'serve-{port}.lock')
        self.fd = None

    def acquire(self, wait=15.0):
        flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(self.path, flags, 0o600)
        deadline = time.monotonic() + wait
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt
                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b'0')
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.fd = fd
                return True
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    os.close(fd)
                    raise
            if time.monotonic() >= deadline:
                os.close(fd)
                return False
            time.sleep(0.2)

    def release(self):
        fd, self.fd = self.fd, None
        if fd is None:
            return
        try:
            if os.name == 'nt':
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)


def service_secret(directory):
    """A per-install secret, readable only by its owner.

    /api/hello proves the service knows it without disclosing it: a process
    of another user squatting the port while it is free can answer the
    hello, but it cannot read this file, so it cannot answer the challenge.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, 'service-secret')
    try:
        with open(path, encoding='ascii') as fh:
            value = fh.read().strip()
        if len(value) >= 32:
            return value
    except OSError:
        pass
    import secrets
    value = secrets.token_hex(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        # Another process created it between our read and now: use theirs.
        with open(path, encoding='ascii') as fh:
            return fh.read().strip()
    with os.fdopen(fd, 'w', encoding='ascii') as fh:
        fh.write(value)
    return value


def hello_proof(secret, nonce):
    import hmac
    # 'devinx-hello' is a protocol constant, kept through the rename: a
    # service from before it must still prove itself to be replaced.
    return hmac.new(secret.encode(), b'devinx-hello\0' + nonce.encode(),
                    hashlib.sha256).hexdigest()
