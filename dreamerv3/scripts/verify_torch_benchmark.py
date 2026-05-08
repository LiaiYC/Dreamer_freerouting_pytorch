import argparse
import pathlib
import re
import subprocess
import sys


def _parse_args():
  parser = argparse.ArgumentParser(
      description='Verify PyTorch Freerouting benchmark output and GPU availability.')
  parser.add_argument('--data-dir', default='')
  parser.add_argument('--manifest', default='')
  parser.add_argument('--random-boards', type=int, default=3)
  parser.add_argument('--epochs', type=int, default=2)
  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--max-boards', type=int, default=128)
  parser.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
  return parser.parse_args()


def main():
  args = _parse_args()
  caller_cwd = pathlib.Path.cwd()

  def _resolve_optional(path_text):
    if not path_text:
      return ''
    path = pathlib.Path(path_text).expanduser()
    if not path.is_absolute():
      path = (caller_cwd / path).resolve()
    return str(path)

  args.data_dir = _resolve_optional(args.data_dir)
  args.manifest = _resolve_optional(args.manifest)

  if not args.data_dir and not args.manifest:
    raise SystemExit('verify_torch_benchmark.py requires --data-dir or --manifest.')

  repo_root = pathlib.Path(__file__).resolve().parents[1]
  launcher = repo_root / 'scripts' / 'train_freerouting.py'

  cmd = [
      sys.executable,
      str(launcher),
      '--framework',
      'pytorch',
      '--torch-epochs',
      str(args.epochs),
      '--torch-batch-size',
      str(args.batch_size),
      '--torch-max-boards',
      str(args.max_boards),
      '--torch-random-boards',
      str(args.random_boards),
      '--device',
      args.device,
  ]
  if args.data_dir:
    cmd += ['--data-dir', args.data_dir]
  if args.manifest:
    cmd += ['--manifest', args.manifest]

  print('Running verification command:')
  print('  ' + ' '.join(cmd))

  result = subprocess.run(
      cmd,
      cwd=str(repo_root),
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      encoding='utf-8',
      errors='replace',
      check=False,
  )
  print(result.stdout)

  if result.returncode != 0:
    raise SystemExit(f'Benchmark command failed with exit code {result.returncode}.')

  expected_source = f'freerouting_boards={args.random_boards}'
  if not re.search(rf'^\s*source\s*:\s*{re.escape(expected_source)}\s*$', result.stdout, re.MULTILINE):
    raise SystemExit(
        f'Missing expected source line: source : {expected_source}.')

  if not re.search(r'^\s*training_status\s*:\s*completed\s*$', result.stdout, re.MULTILINE):
    raise SystemExit('Missing expected line: training_status : completed')

  if args.device == 'gpu':
    if not re.search(r'^\s*resolved_device\s*:\s*cuda\s*$', result.stdout, re.MULTILINE):
      raise SystemExit('GPU verification failed: resolved_device is not cuda.')
    if not re.search(r'^\s*cuda_available\s*:\s*True\s*$', result.stdout, re.MULTILINE):
      raise SystemExit('GPU verification failed: cuda_available is not True.')

  print('Verification passed:')
  print('  - GPU runnable check: passed' if args.device == 'gpu' else '  - CPU run check: passed')
  print(f'  - source check      : passed ({expected_source})')


if __name__ == '__main__':
  main()
