import argparse
import json
import math
import pathlib
import re
import time

import numpy as np


NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')
RES_RE = re.compile(r'\(resolution\s+([A-Za-z]+)\s+([0-9.]+)')
NET_RE = re.compile(r'\(net\s')
COMP_RE = re.compile(r'\(component\s')
PIN_RE = re.compile(r'\(pin\s')
KEEP_RE = re.compile(r'\(keepout\b')
LAYER_RE = re.compile(r'\(layer\s+([^\s\)]+)')

UNIT_TO_MM = {
    'um': 1e-3,
    'mm': 1.0,
    'mil': 0.0254,
}


def _parse_args():
  parser = argparse.ArgumentParser(
      description='Small PyTorch training benchmark for Freerouting features.')
  parser.add_argument('--data-dir', default='')
  parser.add_argument('--manifest', default='')
  parser.add_argument('--patterns', default='*.dsn,*.DSN')
  parser.add_argument('--max-boards', type=int, default=128)
  parser.add_argument('--random-boards', type=int, default=0)
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--hidden-units', type=int, default=64)
  parser.add_argument('--learning-rate', type=float, default=1e-3)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--synthetic-samples', type=int, default=512)
  parser.add_argument('--feature-dim', type=int, default=8)
  parser.add_argument('--target-dim', type=int, default=7)
  parser.add_argument('--steps-per-epoch', type=int, default=0)
  parser.add_argument('--device', choices=('auto', 'cpu', 'gpu'), default='auto')
  return parser.parse_args()


def _ensure_dsn(path, source):
  if path.suffix.lower() != '.dsn':
    raise SystemExit(
        f'Non-DSN file in {source}: {path} (expected .dsn/.DSN).')


def _find_dsn_files(data_dir, patterns):
  files = []
  for pattern in patterns:
    files.extend(data_dir.rglob(pattern))
  return sorted({p.resolve() for p in files})


def _extract_feature_vector(path, feature_dim):
  size_bytes = path.stat().st_size
  num_nets = 0
  num_components = 0
  num_pins = 0
  num_keepouts = 0
  layer_names = set()
  unit = None
  resolution = None

  boundary_numbers = []
  in_boundary = False
  boundary_depth = 0

  with path.open('r', encoding='utf-8', errors='ignore') as f:
    for line in f:
      if unit is None:
        res_match = RES_RE.search(line)
        if res_match:
          unit = res_match.group(1).lower()
          resolution = float(res_match.group(2))
      num_nets += len(NET_RE.findall(line))
      num_components += len(COMP_RE.findall(line))
      num_pins += len(PIN_RE.findall(line))
      num_keepouts += len(KEEP_RE.findall(line))
      for name in LAYER_RE.findall(line):
        layer_names.add(name)

      if '(boundary' in line:
        in_boundary = True
        boundary_depth = line.count('(') - line.count(')')
        boundary_numbers.extend([float(x) for x in NUM_RE.findall(line)])
        if boundary_depth <= 0:
          in_boundary = False
      elif in_boundary:
        boundary_depth += line.count('(') - line.count(')')
        boundary_numbers.extend([float(x) for x in NUM_RE.findall(line)])
        if boundary_depth <= 0:
          in_boundary = False

  width_mm, height_mm = 0.0, 0.0
  if boundary_numbers:
    coords = boundary_numbers[1:] if len(boundary_numbers) > 1 else []
    xs = coords[0::2]
    ys = coords[1::2]
    if xs and ys:
      width_units = max(xs) - min(xs)
      height_units = max(ys) - min(ys)
      scale = resolution or 1.0
      scale *= UNIT_TO_MM.get(unit, 1.0)
      width_mm = width_units * scale
      height_mm = height_units * scale

  values = np.array([
      np.log1p(size_bytes),
      np.log1p(num_nets),
      np.log1p(num_components),
      np.log1p(num_pins),
      np.log1p(len(layer_names)),
      np.log1p(num_keepouts),
      np.log1p(max(width_mm, 0.0)),
      np.log1p(max(height_mm, 0.0)),
  ], np.float32)

  if feature_dim <= len(values):
    return values[:feature_dim]
  pad = np.zeros((feature_dim - len(values),), np.float32)
  return np.concatenate([values, pad], axis=0)


def _build_targets(features, target_dim, seed):
  rng = np.random.default_rng(seed)
  weights = rng.normal(0.0, 0.35, size=(features.shape[1], target_dim)).astype(np.float32)
  bias = rng.normal(0.0, 0.10, size=(target_dim,)).astype(np.float32)
  return np.tanh(features @ weights + bias).astype(np.float32)


def _resolve_device(torch, requested):
  cuda_available = bool(torch.cuda.is_available())
  gpu_name = torch.cuda.get_device_name(0) if cuda_available else 'N/A'
  if requested == 'cpu':
    resolved = 'cpu'
  elif requested == 'gpu':
    resolved = 'cuda' if cuda_available else 'cpu'
  else:
    resolved = 'cuda' if cuda_available else 'cpu'
  return torch.device(resolved), resolved, cuda_available, gpu_name


def _load_features(args):
  boards = []
  source_detail = ''

  if args.manifest:
    manifest = pathlib.Path(args.manifest).expanduser().resolve()
    if not manifest.exists():
      raise FileNotFoundError(manifest)
    base_dir = manifest.parent
    if args.data_dir:
      base_dir = pathlib.Path(args.data_dir).expanduser().resolve()
    source_detail = f'manifest={manifest}'
    with manifest.open('r', encoding='utf-8') as f:
      for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
          continue
        entry = json.loads(line)
        board_text = entry.get('path', '')
        if not board_text:
          raise SystemExit(f'Missing `path` in manifest line {i}: {manifest}')
        board_path = pathlib.Path(board_text)
        if not board_path.is_absolute():
          board_path = (base_dir / board_path).resolve()
        _ensure_dsn(board_path, f'manifest line {i}')
        if not board_path.exists():
          raise FileNotFoundError(board_path)
        feats = entry.get('features', None)
        if feats is None:
          feats = _extract_feature_vector(board_path, args.feature_dim)
        feats = np.asarray(feats, np.float32)
        boards.append(dict(
            path=board_path,
            name=entry.get('name', board_path.stem),
            features=feats,
        ))
  elif args.data_dir:
    data_dir = pathlib.Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
      raise FileNotFoundError(data_dir)
    source_detail = f'data_dir={data_dir}'
    patterns = [p.strip() for p in args.patterns.split(',') if p.strip()]
    files = _find_dsn_files(data_dir, patterns)
    for path in files:
      _ensure_dsn(path, f'data_dir={data_dir}')
      boards.append(dict(
          path=path,
          name=path.stem,
          features=_extract_feature_vector(path, args.feature_dim),
      ))
  else:
    return None, 'synthetic', [], 'synthetic', []

  if args.random_boards > 0:
    if len(boards) < args.random_boards:
      raise SystemExit(
          f'Not enough DSN boards for random sampling: '
          f'found={len(boards)}, requested={args.random_boards}')
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(boards), size=args.random_boards, replace=False)
    boards = [boards[int(i)] for i in idxs]
    source_detail += f', random_boards={args.random_boards}'
  elif args.max_boards > 0 and len(boards) > args.max_boards:
    boards = boards[:args.max_boards]

  if args.max_boards > 0 and len(boards) > args.max_boards:
    boards = boards[:args.max_boards]

  if not boards:
    raise SystemExit(
        'No DSN boards found from provided inputs. '
        f'source={source_detail or "unknown"}, patterns={args.patterns}. '
        'Because --data-dir/--manifest is set, fallback is disabled.')

  feats = np.stack([np.asarray(b['features'], np.float32) for b in boards], axis=0)
  preview = [str(b.get('path')) for b in boards[:3] if b.get('path')]
  selected_names = [str(b.get('name', pathlib.Path(b['path']).stem)) for b in boards]
  return feats, f'freerouting_boards={len(boards)}', preview, source_detail, selected_names


def main():
  args = _parse_args()
  if args.epochs <= 0:
    raise SystemExit('--epochs must be > 0')
  if args.batch_size <= 0:
    raise SystemExit('--batch-size must be > 0')

  try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset
  except Exception as exc:  # pragma: no cover
    raise SystemExit(
        'PyTorch is not installed. Run: pip install -r requirements-pytorch.txt') from exc

  device, resolved_device, cuda_available, gpu_name = _resolve_device(torch, args.device)

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  features, source, preview_paths, source_detail, selected_names = _load_features(args)
  if features is None:
    rng = np.random.default_rng(args.seed)
    features = rng.normal(size=(args.synthetic_samples, args.feature_dim)).astype(np.float32)
    preview_paths = []
    selected_names = []
    source_detail = 'synthetic'
  else:
    print(f'Loaded DSN features from {source_detail}')
    if preview_paths:
      print('Sample boards:')
      for item in preview_paths:
        print(f'  - {item}')
    if args.random_boards > 0:
      print(f'Randomly selected boards ({len(selected_names)}):')
      for name in selected_names:
        print(f'  - {name}')

  mean = features.mean(axis=0, keepdims=True)
  std = features.std(axis=0, keepdims=True) + 1e-6
  features = (features - mean) / std
  targets = _build_targets(features, args.target_dim, args.seed + 1)

  batch_size = min(args.batch_size, len(features))
  steps_per_epoch = args.steps_per_epoch or math.ceil(len(features) / batch_size)
  steps_per_epoch = max(1, int(steps_per_epoch))

  features_t = torch.as_tensor(features, dtype=torch.float32)
  targets_t = torch.as_tensor(targets, dtype=torch.float32)
  dataset = TensorDataset(features_t, targets_t)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

  model = nn.Sequential(
      nn.Linear(features.shape[1], args.hidden_units),
      nn.ReLU(),
      nn.Linear(args.hidden_units, args.hidden_units),
      nn.ReLU(),
      nn.Linear(args.hidden_units, args.target_dim),
  ).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  loss_fn = nn.MSELoss()

  def train_step(batch_x, batch_y):
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    optimizer.zero_grad(set_to_none=True)
    pred = model(batch_x)
    loss = loss_fn(pred, batch_y)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())

  # Warm-up once so startup overhead does not dominate epoch timing.
  warmup_x, warmup_y = next(iter(loader))
  _ = train_step(warmup_x, warmup_y)

  epoch_times = []
  total_start = time.perf_counter()
  for epoch in range(1, args.epochs + 1):
    start = time.perf_counter()
    losses = []
    data_iter = iter(loader)
    for _ in range(steps_per_epoch):
      try:
        batch_x, batch_y = next(data_iter)
      except StopIteration:
        data_iter = iter(loader)
        batch_x, batch_y = next(data_iter)
      losses.append(train_step(batch_x, batch_y))
    elapsed = time.perf_counter() - start
    epoch_times.append(elapsed)
    print(
        f'Epoch {epoch:02d}/{args.epochs} '
        f'- loss: {np.mean(losses):.6f} '
        f'- time: {elapsed:.3f}s')

  total_elapsed = time.perf_counter() - total_start
  print()
  print('PyTorch benchmark summary')
  print(f'  source           : {source}')
  print(f'  source_detail    : {source_detail}')
  print(f'  samples          : {len(features)}')
  print(f'  batch_size       : {batch_size}')
  print(f'  steps_per_epoch  : {steps_per_epoch}')
  print(f'  epochs           : {args.epochs}')
  print(f'  requested_device : {args.device}')
  print(f'  resolved_device  : {resolved_device}')
  print(f'  cuda_available   : {cuda_available}')
  print(f'  gpu_name         : {gpu_name}')
  print(f'  total_time_sec   : {total_elapsed:.3f}')
  print(f'  avg_epoch_sec    : {np.mean(epoch_times):.3f}')
  print(f'  min_epoch_sec    : {np.min(epoch_times):.3f}')
  print(f'  max_epoch_sec    : {np.max(epoch_times):.3f}')
  print('  training_status  : completed')


if __name__ == '__main__':
  main()
