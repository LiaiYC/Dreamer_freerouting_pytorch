import argparse
import os
import pathlib
import subprocess
import sys


def _parse_args():
  parser = argparse.ArgumentParser(
      description='Launch DreamerV3 training on the Freerouting environment.')
  parser.add_argument(
      '--framework', choices=('jax', 'pytorch'), default='jax',
      help='`jax` runs DreamerV3 training; `pytorch` runs PyTorch benchmark or RL probe.')
  parser.add_argument('--jar', default=os.environ.get('FREEROUTING_JAR', ''))
  parser.add_argument('--data-dir', default=os.environ.get('FREEROUTING_DATA_DIR', ''))
  parser.add_argument('--manifest', default=os.environ.get('FREEROUTING_MANIFEST', ''))
  parser.add_argument('--logdir', default='')
  parser.add_argument('--configs', nargs='*', default=['freerouting'])
  parser.add_argument('--task', default='freerouting_route')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--jax-platform', default='')
  parser.add_argument('--torch-epochs', type=int, default=20)
  parser.add_argument('--torch-batch-size', type=int, default=32)
  parser.add_argument('--torch-max-boards', type=int, default=128)
  parser.add_argument('--torch-random-boards', type=int, default=0)
  parser.add_argument('--torch-synthetic-samples', type=int, default=512)
  parser.add_argument('--torch-rl-probe', action='store_true')
  parser.add_argument('--torch-episodes', type=int, default=20)
  parser.add_argument('--torch-timeout', type=float, default=120.0)
  parser.add_argument('--torch-java', default='java')
  parser.add_argument('--torch-save-outputs', action='store_true')
  parser.add_argument('--device', choices=('auto', 'cpu', 'gpu'), default='auto')
  return parser.parse_known_args()


def main():
  args, extra = _parse_args()
  caller_cwd = pathlib.Path.cwd()

  def _resolve_optional(path_text):
    if not path_text:
      return ''
    path = pathlib.Path(path_text).expanduser()
    if not path.is_absolute():
      path = (caller_cwd / path).resolve()
    return str(path)

  args.jar = _resolve_optional(args.jar)
  args.data_dir = _resolve_optional(args.data_dir)
  args.manifest = _resolve_optional(args.manifest)
  args.logdir = _resolve_optional(args.logdir)

  repo_root = pathlib.Path(__file__).resolve().parents[1]
  if args.framework == 'pytorch':
    if args.torch_rl_probe:
      if not args.jar:
        raise SystemExit('Missing --jar or FREEROUTING_JAR for --torch-rl-probe')
      cmd = [sys.executable, str(repo_root / 'scripts' / 'torch_freerouting_rl_probe.py')]
      cmd += ['--jar', args.jar]
      cmd += ['--episodes', str(args.torch_episodes)]
      cmd += ['--timeout', str(args.torch_timeout)]
      cmd += ['--java', args.torch_java]
      cmd += ['--device', args.device]
      if args.torch_random_boards > 0:
        cmd += ['--random-boards', str(args.torch_random_boards)]
      if args.data_dir:
        cmd += ['--data-dir', args.data_dir]
      if args.manifest:
        cmd += ['--manifest', args.manifest]
      if args.torch_save_outputs:
        cmd += ['--save-outputs']
      cmd += extra
    else:
      cmd = [sys.executable, str(repo_root / 'scripts' / 'torch_freerouting_benchmark.py')]
      cmd += ['--epochs', str(args.torch_epochs)]
      cmd += ['--batch-size', str(args.torch_batch_size)]
      cmd += ['--max-boards', str(args.torch_max_boards)]
      cmd += ['--synthetic-samples', str(args.torch_synthetic_samples)]
      cmd += ['--device', args.device]
      if args.torch_random_boards > 0:
        cmd += ['--random-boards', str(args.torch_random_boards)]
      if args.data_dir:
        cmd += ['--data-dir', args.data_dir]
      if args.manifest:
        cmd += ['--manifest', args.manifest]
      cmd += extra
  else:
    if not args.jar:
      raise SystemExit('Missing --jar or FREEROUTING_JAR')
    # data_dir/manifest can come from configs.yaml defaults if not provided here.
    cmd = [sys.executable, str(repo_root / 'dreamerv3' / 'main.py')]
    cmd += ['--configs'] + list(args.configs)
    if args.debug:
      cmd += ['debug']
    cmd += ['--task', args.task]
    cmd += ['--framework=jax']
    cmd += [f'--env.freerouting.jar={args.jar}']
    if args.data_dir:
      cmd += [f'--env.freerouting.data_dir={args.data_dir}']
    if args.manifest:
      cmd += [f'--env.freerouting.manifest={args.manifest}']
    if args.logdir:
      cmd += [f'--logdir={args.logdir}']
    if args.jax_platform:
      cmd += [f'--jax.platform={args.jax_platform}']
    elif args.device in ('cpu', 'gpu'):
      cmd += [f'--jax.platform={"cpu" if args.device == "cpu" else "cuda"}']
    cmd += extra

  print('Running:', ' '.join(cmd), flush=True)
  result = subprocess.run(cmd, check=False, cwd=str(repo_root))
  if result.returncode:
    raise SystemExit(result.returncode)


if __name__ == '__main__':
  main()
