"""Check every test and complete command against an installed wheel outside checkout.

Run with a Python environment containing setuptools and wheel. Temporary build
and install directories are removed on exit. No runtime dependency is needed.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
COMMANDS = [('demo',), ('audit', 'examples/input.json'), ('path-demo',), ('trace', 'examples/path-input.json'), ('curvature', 'examples/curvature-input.json')]


def run(args, cwd):
    return subprocess.check_output(list(map(str, args)), cwd=cwd, env=ENV)


def main():
    expected = [run([sys.executable, '-m', 'influencelab', *args], ROOT) for args in COMMANDS]
    with tempfile.TemporaryDirectory(prefix='influencelab-wheel-') as directory:
        temporary = Path(directory)
        run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
             '--wheel-dir', temporary / 'dist', ROOT], temporary)
        run([sys.executable, '-m', 'venv', temporary / 'env'], temporary)
        python = temporary / 'env' / 'bin' / 'python'
        wheels = list((temporary / 'dist').glob('*.whl'))
        assert len(wheels) == 1
        run([python, '-m', 'pip', 'install', '--no-deps', wheels[0]], temporary)
        outside = temporary / 'outside'
        outside.mkdir()
        for name in ('examples', 'assets', 'tests'):
            shutil.copytree(ROOT / name, outside / name, ignore=shutil.ignore_patterns('__pycache__'))
        check = '''import importlib, pathlib, sys, unittest
names = ['influencelab','influencelab.core','influencelab.path','influencelab.demo','influencelab.__main__', 'influencelab.curvature']
modules = [importlib.import_module(name) for name in names]
def origins():
    for module in modules:
        assert pathlib.Path(module.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()), module.__file__
origins()
sys.path.insert(0, str(pathlib.Path.cwd()))
suite = unittest.defaultTestLoader.discover('tests')
result = unittest.TextTestRunner(verbosity=2).run(suite)
origins()
assert result.wasSuccessful()
print('Full installed suite passed; all six package origins checked before and after')
'''
        print(run([python, '-I', '-c', check], outside).decode(), end='')
        for args, original in zip(COMMANDS, expected):
            assert run([python, '-I', '-m', 'influencelab', *args], outside) == original, args
        for renderer in ('render_demo.py', 'render_paths.py'):
            run([python, '-I', outside / 'examples' / renderer, '--check'], outside)
        print('All five complete module CLI outputs match; both original figures regenerate')


if __name__ == '__main__':
    main()
