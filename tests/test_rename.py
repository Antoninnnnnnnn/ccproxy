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


if __name__ == '__main__':
    unittest.main()
