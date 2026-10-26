import runpy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/train.py'


class LauncherTest(unittest.TestCase):
    def command(self, *args):
        with patch.object(sys, 'argv', [str(SCRIPT), '--out', 'test-output', *args]), \
             patch('subprocess.run') as run:
            runpy.run_path(str(SCRIPT), run_name='__main__')
        return run.call_args.args[0]

    def test_ode_defaults_and_as_forwarding(self):
        cmd = self.command('--method', 'ode', '--solver', 'midpoint', '--saturation-weight', '0.01')
        self.assertTrue(cmd[2].endswith('train_ode.py'))
        for key, value in [('--solver', 'midpoint'), ('--actor-microbatch', '1024'),
                           ('--actor-batch', '4096'), ('--saturation-weight', '0.01')]:
            self.assertEqual(cmd[cmd.index(key)+1], value)

    def test_mlp_defaults_preserved(self):
        cmd = self.command('--method', 'baseline')
        self.assertTrue(cmd[2].endswith('train.py'))
        self.assertEqual(cmd[cmd.index('--actor-microbatch')+1], '512')
        self.assertNotIn('--solver', cmd)


if __name__ == '__main__':
    unittest.main()
