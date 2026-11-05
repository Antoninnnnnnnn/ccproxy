"""devinx became ccproxy: what the old name left on a machine keeps working,
and the move never touches anything but the data directory."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import runtime_support


class RenameTests(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = tmp.name
        env = mock.patch.dict(os.environ, {'XDG_DATA_HOME': self.base,
                                           'LOCALAPPDATA': self.base})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('CCPROXY_DATA', None)
        base = mock.patch.object(runtime_support, '_data_base', lambda: self.base)
        base.start()
        self.addCleanup(base.stop)
        self.old = os.path.join(self.base, 'devinx')
        self.new = os.path.join(self.base, 'ccproxy')

    def legacy(self):
        os.makedirs(os.path.join(self.old, 'kimi'))
        for name, text in (('service-secret', 's'), ('devinx.log', 'history\n')):
            with open(os.path.join(self.old, name), 'w') as fh:
                fh.write(text)

    def test_the_old_directory_is_used_until_it_moves(self):
        self.legacy()
        self.assertEqual(runtime_support.data_dir(), os.path.normpath(self.old))

    def test_the_data_directory_and_its_log_move(self):
        self.legacy()
        here = runtime_support.migrate_data_dir()
        self.assertEqual(os.path.normpath(here), os.path.normpath(self.new))
        with open(os.path.join(self.new, 'ccproxy.log')) as fh:
            self.assertEqual(fh.read(), 'history\n')
        self.assertTrue(os.path.exists(os.path.join(self.new, 'service-secret')))
        self.assertEqual(runtime_support.data_dir(), os.path.normpath(self.new))
        if os.name == 'posix':
            # A devinx service still running reads its files through the links.
            self.assertTrue(os.path.islink(self.old))
            self.assertTrue(os.path.exists(os.path.join(self.old, 'devinx.log')))

    def test_a_checkout_named_devinx_is_never_moved(self):
        # Where LOCALAPPDATA is missing the Windows base is the home directory,
        # and ~/devinx there is the repository.
        os.makedirs(os.path.join(self.old, '.git'))
        with open(os.path.join(self.old, 'service-secret'), 'w') as fh:
            fh.write('looks like data')
        runtime_support.migrate_data_dir()
        self.assertTrue(os.path.isdir(os.path.join(self.old, '.git')))
        self.assertFalse(os.path.exists(self.new))
        self.assertEqual(runtime_support.data_dir(), os.path.normpath(self.new))

    def test_an_unrelated_devinx_directory_is_left_alone(self):
        os.makedirs(os.path.join(self.old, 'src'))
        runtime_support.migrate_data_dir()
        self.assertTrue(os.path.isdir(os.path.join(self.old, 'src')))
        self.assertFalse(os.path.exists(self.new))

    def test_legacy_variables_are_read_under_the_new_name(self):
        env = {'DEVINX_SWE': '1', 'DEVINX_PORT': '9000', 'CCPROXY_PORT': '8316'}
        runtime_support.mirror_legacy_env(env)
        self.assertEqual(env['CCPROXY_SWE'], '1')
        self.assertEqual(env['CCPROXY_PORT'], '8316')   # the new name wins



class LegacyServiceTests(unittest.TestCase):
    """A devinx service still running must be judged stale and replaced, not
    taken for a stranger: both of these were broken by the rename itself."""

    def test_the_identity_proof_is_unchanged(self):
        import hmac, hashlib
        expected = hmac.new(b's', b'devinx-hello\0' + b'ab12',
                            hashlib.sha256).hexdigest()
        self.assertEqual(runtime_support.hello_proof('s', 'ab12'), expected)

    @unittest.skipUnless(sys.platform.startswith('linux'), '/proc cmdline')
    def test_a_devinx_process_may_be_stopped(self):
        import launcher
        data = b'/venv/bin/python\0/home/u/devinx/devinx.py\0'
        with mock.patch('builtins.open', mock.mock_open(read_data=data)):
            self.assertTrue(launcher._pid_runs_ccproxy(1234))
        with mock.patch('builtins.open', mock.mock_open(read_data=b'/usr/bin/vim\0')):
            self.assertFalse(launcher._pid_runs_ccproxy(1234))

    def test_conversation_ids_keep_their_seed(self):
        # SWE-2 cascade and message ids derive from these: a new seed would
        # start every conversation in flight over upstream.
        source = (ROOT / 'ccproxy.py').read_text(encoding='utf-8')
        for seed in ('"devinx-msg\\0"', '"devinx\\0" + conv_seed', '"devinx-traj\\0"'):
            self.assertIn(seed, source)

if __name__ == '__main__':
    unittest.main()
