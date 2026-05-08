import argparse
import json
import math
import pathlib
import re
import subprocess
import tempfile
import time

import numpy as np


NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')
RES_RE = re.compile(r'\(resolution\s+([A-Za-z]+)\s+([0-9.]+)')
NET_RE = re.compile(r'\(net\s')
COMP_RE = re.compile(r'\(component\s')
PIN_RE = re.compile(r'\(pin\s')
KEEP_RE = re.compile(r'\(keepout\b')
LAYER_RE = re.compile(r'\(layer\s+([^\s\)]+)')
NET_START_RE = re.compile(r'\(net\s+([^\s\)]+)')

UNIT_TO_MM = {
    'um': 1e-3,
    'mm': 1.0,
    'mil': 0.0254,
}

PARAM_SPECS = [
    dict(name='router.max_passes', low=20.0, high=300.0, kind='int'),
    dict(name='router.via_costs', low=1.0, high=200.0, kind='float'),
    dict(name='router.plane_via_costs', low=1.0, high=80.0, kind='float'),
    dict(name='router.start_ripup_costs', low=10.0, high=400.0, kind='float'),
    dict(name='router.default_preferred_direction_trace_cost', low=0.5, high=4.0, kind='float'),
    dict(name='router.default_undesired_direction_trace_cost', low=1.0, high=8.0, kind='float'),
]

REWARD = dict(
    length=0.001,
    vias=1.0,
    violations=5.0,
    unconnected=10.0,
    success=50.0,
    fail=-50.0,
)


def _parse_args():
  parser = argparse.ArgumentParser(
      description='PyTorch RL probe that interacts with Freerouting directly.')
  parser.add_argument('--jar', required=True)
  parser.add_argument('--data-dir', default='')
  parser.add_argument('--manifest', default='')
  parser.add_argument('--patterns', default='*.dsn,*.DSN')
  parser.add_argument('--random-boards', type=int, default=3)
  parser.add_argument('--episodes', type=int, default=20)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--java', default='java')
  parser.add_argument('--timeout', type=float, default=120.0)
  parser.add_argument('--threads', type=int, default=1)
  parser.add_argument('--save-outputs', action='store_true')
  parser.add_argument('--workdir', default='')
  parser.add_argument('--device', choices=('auto', 'cpu', 'gpu'), default='auto')
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--entropy-coef', type=float, default=1e-3)
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


def _path_length(nums):
  if len(nums) < 5:
    return 0.0
  coords = nums[1:]
  if len(coords) < 4:
    return 0.0
  total = 0.0
  x_prev, y_prev = coords[0], coords[1]
  for i in range(2, len(coords) - 1, 2):
    x, y = coords[i], coords[i + 1]
    dx = x - x_prev
    dy = y - y_prev
    total += math.sqrt(dx * dx + dy * dy)
    x_prev, y_prev = x, y
  return total


def parse_ses_metrics(path):
  path = pathlib.Path(path)
  via_count = 0
  length_units = 0.0
  unit = None
  resolution = None
  net_name = None
  net_depth = 0
  net_lengths_units = {}
  net_vias = {}

  in_path = False
  path_depth = 0
  path_numbers = []

  with path.open('r', encoding='utf-8', errors='ignore') as f:
    for line in f:
      if unit is None:
        res_match = RES_RE.search(line)
        if res_match:
          unit = res_match.group(1).lower()
          resolution = float(res_match.group(2))

      net_match = NET_START_RE.search(line)
      delta = line.count('(') - line.count(')')
      if net_match:
        net_name = net_match.group(1)
        net_depth = delta
      elif net_name is not None:
        net_depth += delta

      if '(via' in line:
        count = line.count('(via')
        via_count += count
        if net_name:
          net_vias[net_name] = net_vias.get(net_name, 0) + count

      if '(path' in line:
        in_path = True
        path_depth = line.count('(') - line.count(')')
        path_numbers.extend([float(x) for x in NUM_RE.findall(line)])
        if path_depth <= 0:
          plen = _path_length(path_numbers)
          length_units += plen
          if net_name:
            net_lengths_units[net_name] = net_lengths_units.get(net_name, 0.0) + plen
          path_numbers = []
          in_path = False
      elif in_path:
        path_depth += line.count('(') - line.count(')')
        path_numbers.extend([float(x) for x in NUM_RE.findall(line)])
        if path_depth <= 0:
          plen = _path_length(path_numbers)
          length_units += plen
          if net_name:
            net_lengths_units[net_name] = net_lengths_units.get(net_name, 0.0) + plen
          path_numbers = []
          in_path = False

      if net_name is not None and net_depth <= 0:
        net_name = None

  scale = (resolution or 1.0) * UNIT_TO_MM.get(unit or '', 1.0)
  net_lengths_mm = {k: v * scale for k, v in net_lengths_units.items()}
  return dict(
      length_mm=length_units * scale,
      vias=via_count,
      net_lengths_mm=net_lengths_mm,
      net_vias=net_vias,
  )


def parse_drc_metrics(drc):
  violations = drc.get('violations', []) if isinstance(drc, dict) else []
  unconnected = drc.get('unconnected_items', []) if isinstance(drc, dict) else []
  return dict(
      violations=float(len(violations)),
      unconnected=float(len(unconnected)),
  )


def extract_board_features(path):
  path = pathlib.Path(path)
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
      scale = (resolution or 1.0) * UNIT_TO_MM.get(unit or '', 1.0)
      width_mm = width_units * scale
      height_mm = height_units * scale

  features = np.array([
      np.log1p(size_bytes),
      np.log1p(num_nets),
      np.log1p(num_components),
      np.log1p(num_pins),
      np.log1p(len(layer_names)),
      np.log1p(num_keepouts),
      np.log1p(max(width_mm, 0.0)),
      np.log1p(max(height_mm, 0.0)),
  ], np.float32)
  return features


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


def _load_boards(args):
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
        board = pathlib.Path(board_text)
        if not board.is_absolute():
          board = (base_dir / board).resolve()
        _ensure_dsn(board, f'manifest line {i}')
        if not board.exists():
          raise FileNotFoundError(board)
        boards.append(board)
  else:
    if not args.data_dir:
      raise SystemExit('Need --data-dir or --manifest for Freerouting RL probe.')
    data_dir = pathlib.Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
      raise FileNotFoundError(data_dir)
    source_detail = f'data_dir={data_dir}'
    patterns = [p.strip() for p in args.patterns.split(',') if p.strip()]
    boards = _find_dsn_files(data_dir, patterns)

  if not boards:
    raise SystemExit(
        'No DSN boards found from provided inputs. '
        f'source={source_detail or "unknown"}, patterns={args.patterns}. '
        'Because --data-dir/--manifest is set, fallback is disabled.')

  rng = np.random.default_rng(args.seed)
  if args.random_boards > 0:
    if len(boards) < args.random_boards:
      raise SystemExit(
          f'Not enough DSN boards: found={len(boards)}, requested={args.random_boards}')
    idxs = rng.choice(len(boards), size=args.random_boards, replace=False)
    boards = [boards[int(i)] for i in idxs]
    source_detail += f', random_boards={args.random_boards}'

  return boards, source_detail


def _build_action_from_u(u, low, high, specs):
  bounded = 1.0 / (1.0 + np.exp(-u))
  raw = low + bounded * (high - low)
  out = []
  for x, spec in zip(raw, specs):
    val = float(np.clip(x, spec['low'], spec['high']))
    if spec['kind'] == 'int':
      val = int(round(val))
    out.append(val)
  return out


def _run_freerouting_once(args, board_path, params, outdir, run_id):
  ses_path = outdir / f'{run_id}.ses'
  drc_path = outdir / f'{run_id}_drc.json'
  route_cmd = [
      args.java,
      '-jar',
      str(args.jar),
      '--gui.enabled=false',
      '--usage_and_diagnostic_data.disable_analytics=true',
      '-de',
      str(board_path),
      '-do',
      str(ses_path),
      f'--router.max_threads={int(args.threads)}',
  ]
  for spec, value in zip(PARAM_SPECS, params):
    route_cmd.append(f'--{spec["name"]}={value}')

  start = time.perf_counter()
  try:
    route = subprocess.run(
        route_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=args.timeout,
        check=False,
    )
  except subprocess.TimeoutExpired:
    elapsed = time.perf_counter() - start
    return dict(reward=REWARD['fail'], success=0.0, failed=1.0, runtime=elapsed, timeout=1.0)

  if route.returncode != 0 or not ses_path.exists():
    elapsed = time.perf_counter() - start
    return dict(reward=REWARD['fail'], success=0.0, failed=1.0, runtime=elapsed)

  drc_cmd = [
      args.java,
      '-jar',
      str(args.jar),
      '--gui.enabled=false',
      '--usage_and_diagnostic_data.disable_analytics=true',
      '-de',
      f'{board_path}+{ses_path}',
      '-drc',
      str(drc_path),
  ]
  try:
    drc = subprocess.run(
        drc_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=args.timeout,
        check=False,
    )
  except subprocess.TimeoutExpired:
    elapsed = time.perf_counter() - start
    return dict(reward=REWARD['fail'], success=0.0, failed=1.0, runtime=elapsed, timeout=1.0)

  elapsed = time.perf_counter() - start
  if drc.returncode != 0 or not drc_path.exists():
    return dict(reward=REWARD['fail'], success=0.0, failed=1.0, runtime=elapsed)

  ses_metrics = parse_ses_metrics(ses_path)
  with drc_path.open('r', encoding='utf-8') as f:
    drc_json = json.load(f)
  drc_metrics = parse_drc_metrics(drc_json)

  length_mm = float(ses_metrics.get('length_mm', 0.0))
  vias = float(ses_metrics.get('vias', 0.0))
  violations = float(drc_metrics.get('violations', 0.0))
  unconnected = float(drc_metrics.get('unconnected', 0.0))

  success = 1.0 if (violations == 0.0 and unconnected == 0.0) else 0.0
  reward = (
      -length_mm * REWARD['length']
      -vias * REWARD['vias']
      -violations * REWARD['violations']
      -unconnected * REWARD['unconnected']
      + (REWARD['success'] if success else 0.0)
  )

  if not args.save_outputs:
    for path in (ses_path, drc_path):
      if path.exists():
        path.unlink()

  return dict(
      reward=reward,
      success=success,
      failed=0.0,
      runtime=elapsed,
      length_mm=length_mm,
      vias=vias,
      violations=violations,
      unconnected=unconnected,
  )


def main():
  args = _parse_args()

  args.jar = str(pathlib.Path(args.jar).expanduser().resolve())
  if not pathlib.Path(args.jar).exists():
    raise SystemExit(f'Freerouting JAR not found: {args.jar}')
  if args.episodes <= 0:
    raise SystemExit('--episodes must be > 0')

  try:
    import torch
    from torch import nn
  except Exception as exc:  # pragma: no cover
    raise SystemExit(
        'PyTorch is not installed. Run: pip install -r requirements-pytorch.txt') from exc

  device, resolved_device, cuda_available, gpu_name = _resolve_device(torch, args.device)

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  rng = np.random.default_rng(args.seed)

  boards, source_detail = _load_boards(args)
  board_features = [extract_board_features(p) for p in boards]

  low = np.array([p['low'] for p in PARAM_SPECS], np.float32)
  high = np.array([p['high'] for p in PARAM_SPECS], np.float32)
  dim = len(PARAM_SPECS)

  model = nn.Sequential(
      nn.Linear(board_features[0].shape[0], 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, dim),
  ).to(device)
  log_std = nn.Parameter(torch.full((dim,), -0.2, device=device))
  optimizer = torch.optim.Adam(list(model.parameters()) + [log_std], lr=args.lr)

  outdir = pathlib.Path(args.workdir).expanduser().resolve() if args.workdir else (
      pathlib.Path(tempfile.gettempdir()) / 'freerouting_rl_probe')
  outdir.mkdir(parents=True, exist_ok=True)

  print(f'Using {len(boards)} boards for RL probe:')
  for board in boards:
    print(f'  - {board}')
  print(f'Random board names: {[p.stem for p in boards]}')

  baseline = 0.0
  rewards = []
  success_count = 0
  start = time.perf_counter()

  for ep in range(1, args.episodes + 1):
    idx = int(rng.integers(len(boards)))
    board = boards[idx]
    obs = torch.as_tensor(board_features[idx], dtype=torch.float32, device=device).unsqueeze(0)

    mean = model(obs)[0]
    std = torch.exp(log_std)
    eps = torch.randn((dim,), device=device)
    u = mean + std * eps

    logp = -0.5 * (((u - mean) / std) ** 2 + 2.0 * torch.log(std) + math.log(2.0 * math.pi))
    logp_sum = torch.sum(logp)
    entropy = torch.sum(0.5 * torch.log(2.0 * math.pi * math.e * (std ** 2)))

    params = _build_action_from_u(u.detach().cpu().numpy(), low, high, PARAM_SPECS)
    metrics = _run_freerouting_once(args, board, params, outdir, f'ep{ep:04d}')
    reward = float(metrics['reward'])
    baseline = 0.9 * baseline + 0.1 * reward
    advantage = reward - baseline

    loss = -float(advantage) * logp_sum - args.entropy_coef * entropy
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    rewards.append(reward)
    success_count += int(metrics.get('success', 0.0) > 0.5)
    print(
        f'Episode {ep:02d}/{args.episodes} '
        f'board={board.name} reward={reward:.3f} '
        f'success={metrics.get("success", 0.0):.0f} '
        f'failed={metrics.get("failed", 0.0):.0f} '
        f'timeout={metrics.get("timeout", 0.0):.0f} '
        f'viol={metrics.get("violations", -1):.0f} '
        f'unconn={metrics.get("unconnected", -1):.0f} '
        f'runtime={metrics.get("runtime", 0.0):.2f}s')

  elapsed = time.perf_counter() - start
  print()
  print('Freerouting RL probe summary (PyTorch)')
  print(f'  source           : freerouting_boards={len(boards)}')
  print(f'  source_detail    : {source_detail}')
  print(f'  episodes         : {args.episodes}')
  print(f'  interactions     : {args.episodes}')
  print(f'  success_count    : {success_count}')
  print(f'  avg_reward       : {float(np.mean(rewards)):.3f}')
  print(f'  requested_device : {args.device}')
  print(f'  resolved_device  : {resolved_device}')
  print(f'  cuda_available   : {cuda_available}')
  print(f'  gpu_name         : {gpu_name}')
  print(f'  total_time_sec   : {elapsed:.3f}')
  print('  training_status  : completed')


if __name__ == '__main__':
  main()
