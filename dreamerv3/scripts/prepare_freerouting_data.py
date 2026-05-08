import argparse
import json
import pathlib
import random
import sys


def _parse_args():
  parser = argparse.ArgumentParser(
      description='Prepare Freerouting DSN manifest with precomputed features.')
  parser.add_argument('--input-dir', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--patterns', default='*.dsn')
  parser.add_argument('--max-files', type=int, default=0)
  parser.add_argument('--shuffle', action='store_true')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--absolute', action='store_true')
  parser.add_argument('--nonrecursive', action='store_true')
  return parser.parse_args()


def main():
  args = _parse_args()
  repo_root = pathlib.Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(repo_root))

  from embodied.envs import freerouting as fr  # noqa: E402

  input_dir = pathlib.Path(args.input_dir).expanduser().resolve()
  if not input_dir.exists():
    raise FileNotFoundError(input_dir)

  patterns = [p.strip() for p in args.patterns.split(',') if p.strip()]
  files = []
  for pattern in patterns:
    if args.nonrecursive:
      files.extend(input_dir.glob(pattern))
    else:
      files.extend(input_dir.rglob(pattern))
  files = sorted({p.resolve() for p in files})

  if args.shuffle:
    rng = random.Random(args.seed)
    rng.shuffle(files)

  if args.max_files and len(files) > args.max_files:
    files = files[:args.max_files]

  output = pathlib.Path(args.output).expanduser()
  output.parent.mkdir(parents=True, exist_ok=True)

  with output.open('w', encoding='utf-8') as f:
    for path in files:
      features, stats = fr.extract_board_features(path)
      record_path = path
      if not args.absolute:
        try:
          record_path = path.relative_to(input_dir)
        except ValueError:
          record_path = path
      record = dict(
          path=str(record_path),
          name=path.stem,
          features=features.tolist(),
      )
      record.update(stats)
      f.write(json.dumps(record) + '\n')

  print(f'Wrote {len(files)} boards to {output}')


if __name__ == '__main__':
  main()
