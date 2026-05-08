import collections
import json
import os
import pathlib
import re
import subprocess
import tempfile
import time

import elements
import embodied
import numpy as np


NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')
RES_RE = re.compile(r'\(resolution\s+([A-Za-z]+)\s+([0-9.]+)')
NET_RE = re.compile(r'\(net\s')
NET_START_RE = re.compile(r'\(net\s+([^\s\)]+)')
COMP_RE = re.compile(r'\(component\s')
PIN_RE = re.compile(r'\(pin\s')
KEEP_RE = re.compile(r'\(keepout\b')
LAYER_RE = re.compile(r'\(layer\s+([^\s\)]+)')

UNIT_TO_MM = {
    'um': 1e-3,
    'mm': 1.0,
    'mil': 0.0254,
}


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
      scale = resolution or 1.0
      scale *= UNIT_TO_MM.get(unit, 1.0)
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

  stats = dict(
      size_bytes=size_bytes,
      nets=num_nets,
      components=num_components,
      pins=num_pins,
      layers=len(layer_names),
      keepouts=num_keepouts,
      width_mm=width_mm,
      height_mm=height_mm,
      unit=unit or '',
      resolution=resolution or 0.0,
  )
  return features, stats


def load_manifest(path, base_dir=None):
  path = pathlib.Path(path)
  base_dir = pathlib.Path(base_dir) if base_dir else path.parent
  boards = []
  with path.open('r', encoding='utf-8') as f:
    for line in f:
      if not line.strip():
        continue
      entry = json.loads(line)
      board_path = pathlib.Path(entry['path'])
      if not board_path.is_absolute():
        board_path = (base_dir / board_path).resolve()
      features = np.asarray(entry['features'], np.float32)
      boards.append(dict(
          path=board_path,
          name=entry.get('name', board_path.stem),
          features=features,
          stats=entry,
      ))
  return boards


def find_dsn_files(data_dir, patterns):
  data_dir = pathlib.Path(data_dir)
  files = []
  for pattern in patterns:
    files.extend(data_dir.rglob(pattern))
  files = sorted({p.resolve() for p in files})
  return files


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
    total += (dx * dx + dy * dy) ** 0.5
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
  net_lengths_units = collections.defaultdict(float)
  net_vias = collections.defaultdict(int)

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
          net_vias[net_name] += count

      if '(path' in line:
        in_path = True
        path_depth = line.count('(') - line.count(')')
        path_numbers.extend([float(x) for x in NUM_RE.findall(line)])
        if path_depth <= 0:
          path_len = _path_length(path_numbers)
          length_units += path_len
          if net_name:
            net_lengths_units[net_name] += path_len
          path_numbers = []
          in_path = False
      elif in_path:
        path_depth += line.count('(') - line.count(')')
        path_numbers.extend([float(x) for x in NUM_RE.findall(line)])
        if path_depth <= 0:
          path_len = _path_length(path_numbers)
          length_units += path_len
          if net_name:
            net_lengths_units[net_name] += path_len
          path_numbers = []
          in_path = False

      if net_name is not None and net_depth <= 0:
        net_name = None

  scale = resolution or 1.0
  scale *= UNIT_TO_MM.get(unit, 1.0)
  length_mm = length_units * scale
  net_lengths_mm = {k: v * scale for k, v in net_lengths_units.items()}
  return dict(
      length_mm=length_mm,
      vias=via_count,
      net_lengths_mm=net_lengths_mm,
      net_vias=dict(net_vias),
      unit=unit or '',
      resolution=resolution or 0.0,
  )


def parse_drc_metrics(drc):
  violations = drc.get('violations', []) if isinstance(drc, dict) else []
  unconnected = drc.get('unconnected_items', []) if isinstance(drc, dict) else []
  by_type = collections.Counter()
  for violation in violations:
    if isinstance(violation, dict):
      vtype = violation.get('type', '')
      if vtype:
        by_type[vtype] += 1
  return dict(
      violation_count=len(violations),
      unconnected_nets=len(unconnected),
      by_type=by_type,
  )


class Freerouting(embodied.Env):

  def __init__(
      self,
      task,
      jar,
      data_dir='',
      manifest='',
      patterns=('*.dsn',),
      timeout=120,
      shuffle=True,
      max_boards=0,
      seed=0,
      reward=None,
      save_outputs=False,
      logdir=None,
      threads=1,
      critical_nets=None,
      java='java',
  ):
    del task
    if not jar:
      raise ValueError('freerouting jar path is required.')
    self._jar = pathlib.Path(jar).expanduser()
    if not self._jar.exists():
      raise FileNotFoundError(self._jar)

    self._data_dir = pathlib.Path(data_dir).expanduser() if data_dir else None
    self._manifest = pathlib.Path(manifest).expanduser() if manifest else None
    self._patterns = tuple(patterns)
    self._timeout = float(timeout)
    self._shuffle = bool(shuffle)
    self._max_boards = int(max_boards) if max_boards else 0
    self._rng = np.random.default_rng(seed)
    self._save_outputs = bool(save_outputs)
    self._threads = int(threads) if threads is not None else 1
    self._java = str(java)

    reward_defaults = dict(
        length=0.001,
        vias=1.0,
        violations=5.0,
        unconnected=10.0,
        success=50.0,
        fail=-50.0,
        runtime=0.0,
        completion=0.0,
        drc={},
        critical_length=0.0,
        critical_vias=0.0,
    )
    if reward:
      reward_defaults.update(reward)
    drc_defaults = dict(
        clearance=0.0,
        hole_clearance=0.0,
        track_dangling=0.0,
        via_dangling=0.0,
    )
    if isinstance(reward_defaults.get('drc'), dict):
      drc_defaults.update(reward_defaults['drc'])
    reward_defaults['drc'] = drc_defaults
    self._reward = reward_defaults

    critical_nets = critical_nets or []
    if isinstance(critical_nets, str):
      critical_nets = [p.strip() for p in critical_nets.split(',') if p.strip()]
    self._critical_patterns = [re.compile(p) for p in critical_nets]

    self._output_dir = None
    if logdir:
      self._output_dir = pathlib.Path(logdir) / 'freerouting'
    else:
      self._output_dir = pathlib.Path(tempfile.gettempdir()) / 'freerouting_env'
    self._output_dir.mkdir(parents=True, exist_ok=True)
    self._run_prefix = f'{os.getpid()}_{int(time.time() * 1e6)}'

    self._boards = self._load_boards()
    if not self._boards:
      raise ValueError('No DSN boards found for freerouting environment.')
    self._feat_dim = len(self._boards[0]['features'])

    self._param_specs = [
        dict(name='router.max_passes', low=10.0, high=200.0, kind='int'),
        dict(name='router.via_costs', low=1.0, high=200.0, kind='float'),
        dict(name='router.plane_via_costs', low=1.0, high=50.0, kind='float'),
        dict(name='router.improvement_threshold', low=0.0, high=0.2, kind='float'),
        dict(name='router.start_ripup_costs', low=10.0, high=300.0, kind='float'),
        dict(name='router.default_preferred_direction_trace_cost', low=0.5, high=3.0, kind='float'),
        dict(name='router.default_undesired_direction_trace_cost', low=1.0, high=5.0, kind='float'),
    ]
    self._act_low = np.array([p['low'] for p in self._param_specs], np.float32)
    self._act_high = np.array([p['high'] for p in self._param_specs], np.float32)

    self._episode = 0
    self._done = True
    self._needs_action = True
    self._current = None

  @property
  def obs_space(self):
    return {
        'vector': elements.Space(np.float32, (self._feat_dim,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/length_mm': elements.Space(np.float32),
        'log/vias': elements.Space(np.float32),
        'log/violations': elements.Space(np.float32),
        'log/violations_clearance': elements.Space(np.float32),
        'log/violations_hole_clearance': elements.Space(np.float32),
        'log/violations_track_dangling': elements.Space(np.float32),
        'log/violations_via_dangling': elements.Space(np.float32),
        'log/unconnected': elements.Space(np.float32),
        'log/completion': elements.Space(np.float32),
        'log/success': elements.Space(np.float32),
        'log/failed': elements.Space(np.float32),
        'log/runtime': elements.Space(np.float32),
        'log/critical_length_mm': elements.Space(np.float32),
        'log/critical_vias': elements.Space(np.float32),
    }

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'params': elements.Space(np.float32, self._act_low.shape, self._act_low, self._act_high),
    }

  def step(self, action):
    reset = bool(action.get('reset'))
    if reset or self._done:
      self._current = self._next_board()
      self._done = False
      self._needs_action = True
      return self._obs(
          reward=0.0,
          is_first=True,
          is_last=False,
          is_terminal=False,
          length_mm=0.0,
          vias=0.0,
          violations=0.0,
          violations_clearance=0.0,
          violations_hole_clearance=0.0,
          violations_track_dangling=0.0,
          violations_via_dangling=0.0,
          unconnected=0.0,
          completion=0.0,
          success=0.0,
          failed=0.0,
          runtime=0.0,
          critical_length_mm=0.0,
          critical_vias=0.0,
      )

    if not self._needs_action:
      return self._obs(
          reward=0.0,
          is_first=False,
          is_last=True,
          is_terminal=True,
          length_mm=0.0,
          vias=0.0,
          violations=0.0,
          violations_clearance=0.0,
          violations_hole_clearance=0.0,
          violations_track_dangling=0.0,
          violations_via_dangling=0.0,
          unconnected=0.0,
          completion=0.0,
          success=0.0,
          failed=1.0,
          runtime=0.0,
          critical_length_mm=0.0,
          critical_vias=0.0,
      )

    params = np.asarray(action['params'], np.float32)
    metrics = self._run_routing(params)
    self._done = True
    self._needs_action = False
    self._episode += 1
    return self._obs(
        reward=metrics['reward'],
        is_first=False,
        is_last=True,
        is_terminal=True,
        length_mm=metrics['length_mm'],
        vias=metrics['vias'],
        violations=metrics['violations'],
        violations_clearance=metrics['violations_clearance'],
        violations_hole_clearance=metrics['violations_hole_clearance'],
        violations_track_dangling=metrics['violations_track_dangling'],
        violations_via_dangling=metrics['violations_via_dangling'],
        unconnected=metrics['unconnected'],
        completion=metrics['completion'],
        success=metrics['success'],
        failed=metrics['failed'],
        runtime=metrics['runtime'],
        critical_length_mm=metrics['critical_length_mm'],
        critical_vias=metrics['critical_vias'],
    )

  def _obs(
      self,
      reward,
      is_first,
      is_last,
      is_terminal,
      length_mm,
      vias,
      violations,
      violations_clearance,
      violations_hole_clearance,
      violations_track_dangling,
      violations_via_dangling,
      unconnected,
      completion,
      success,
      failed,
      runtime,
      critical_length_mm,
      critical_vias,
  ):
    return dict(
        vector=self._current['features'].astype(np.float32),
        reward=np.float32(reward),
        is_first=bool(is_first),
        is_last=bool(is_last),
        is_terminal=bool(is_terminal),
        **{
            'log/length_mm': np.float32(length_mm),
            'log/vias': np.float32(vias),
            'log/violations': np.float32(violations),
            'log/violations_clearance': np.float32(violations_clearance),
            'log/violations_hole_clearance': np.float32(violations_hole_clearance),
            'log/violations_track_dangling': np.float32(violations_track_dangling),
            'log/violations_via_dangling': np.float32(violations_via_dangling),
            'log/unconnected': np.float32(unconnected),
            'log/completion': np.float32(completion),
            'log/success': np.float32(success),
            'log/failed': np.float32(failed),
            'log/runtime': np.float32(runtime),
            'log/critical_length_mm': np.float32(critical_length_mm),
            'log/critical_vias': np.float32(critical_vias),
        },
    )

  def _load_boards(self):
    boards = []
    if self._manifest:
      base_dir = self._data_dir if self._data_dir else self._manifest.parent
      boards = load_manifest(self._manifest, base_dir=base_dir)
      if not boards:
        raise ValueError(
            f'No boards found in manifest: {self._manifest}. '
            'When manifest/data_dir is specified, fallback is disabled.')
      for board in boards:
        path = pathlib.Path(board['path'])
        if path.suffix.lower() != '.dsn':
          raise ValueError(
              f'Non-DSN board path in manifest: {path}. '
              'Expected .dsn/.DSN only.')
        if not path.exists():
          raise FileNotFoundError(path)
    else:
      if not self._data_dir:
        raise ValueError('data_dir or manifest is required for freerouting env.')
      for path in find_dsn_files(self._data_dir, self._patterns):
        if path.suffix.lower() != '.dsn':
          raise ValueError(
              f'Non-DSN board path found from data_dir: {path}. '
              'Expected .dsn/.DSN only.')
        features, stats = extract_board_features(path)
        boards.append(dict(
            path=path,
            name=path.stem,
            features=features,
            stats=stats,
        ))
      if not boards:
        raise ValueError(
            f'No DSN boards found in data_dir={self._data_dir} '
            f'patterns={self._patterns}. Fallback is disabled.')

    if self._max_boards and len(boards) > self._max_boards:
      if self._shuffle:
        indices = self._rng.choice(len(boards), self._max_boards, replace=False)
        boards = [boards[i] for i in indices]
      else:
        boards = boards[:self._max_boards]
    return boards

  def _next_board(self):
    if self._shuffle:
      idx = int(self._rng.integers(len(self._boards)))
    else:
      idx = self._episode % len(self._boards)
    return self._boards[idx]

  def _run_routing(self, params):
    start = time.time()
    failed = 0.0
    success = 0.0
    length_mm = 0.0
    vias = 0.0
    net_lengths_mm = {}
    net_vias = {}
    violations = 0.0
    violations_clearance = 0.0
    violations_hole_clearance = 0.0
    violations_track_dangling = 0.0
    violations_via_dangling = 0.0
    unconnected = 0.0
    completion = 0.0
    critical_length_mm = 0.0
    critical_vias = 0.0

    base = f'{self._run_prefix}_ep{self._episode:06d}'
    ses_path = self._output_dir / f'{base}.ses'
    drc_path = self._output_dir / f'{base}_drc.json'

    route_cmd = [
        self._java,
        '-jar',
        str(self._jar),
        '--gui.enabled=false',
        '--usage_and_diagnostic_data.disable_analytics=true',
        '-de',
        str(self._current['path']),
        '-do',
        str(ses_path),
    ]
    if self._threads is not None and self._threads >= 0:
      route_cmd.append(f'--router.max_threads={int(self._threads)}')

    for spec, raw in zip(self._param_specs, params):
      val = float(np.clip(raw, spec['low'], spec['high']))
      if spec['kind'] == 'int':
        val = int(round(val))
      route_cmd.append(f"--{spec['name']}={val}")

    try:
      route_result = subprocess.run(
          route_cmd,
          cwd=str(self._output_dir),
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
          timeout=self._timeout,
          check=False,
      )
      if route_result.returncode != 0 or not ses_path.exists():
        failed = 1.0
      else:
        ses_metrics = parse_ses_metrics(ses_path)
        length_mm = ses_metrics['length_mm']
        vias = ses_metrics['vias']
        net_lengths_mm = ses_metrics.get('net_lengths_mm', {})
        net_vias = ses_metrics.get('net_vias', {})

      if not failed:
        drc_cmd = [
            self._java,
            '-jar',
            str(self._jar),
            '--gui.enabled=false',
            '--usage_and_diagnostic_data.disable_analytics=true',
            '-de',
            f"{self._current['path']}+{ses_path}",
            '-drc',
            str(drc_path),
        ]
        drc_result = subprocess.run(
            drc_cmd,
            cwd=str(self._output_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self._timeout,
            check=False,
        )
        if drc_result.returncode == 0 and drc_path.exists():
          with drc_path.open('r', encoding='utf-8') as f:
            drc = json.load(f)
          drc_metrics = parse_drc_metrics(drc)
          unconnected = float(drc_metrics['unconnected_nets'])
          violations = float(drc_metrics['violation_count'])
          by_type = drc_metrics['by_type']
          violations_clearance = float(by_type.get('clearance', 0))
          violations_hole_clearance = float(by_type.get('hole_clearance', 0))
          violations_track_dangling = float(by_type.get('track_dangling', 0))
          violations_via_dangling = float(by_type.get('via_dangling', 0))
        else:
          failed = 1.0
      success = 1.0 if (unconnected == 0 and violations == 0 and not failed) else 0.0
    except Exception:
      failed = 1.0
    finally:
      if not self._save_outputs:
        for path in (ses_path, drc_path):
          try:
            if path.exists():
              path.unlink()
          except OSError:
            pass

    runtime = time.time() - start
    if not failed:
      total_nets = float(self._current['stats'].get('nets', 0))
      if total_nets > 0:
        completion = max(0.0, 1.0 - (unconnected / total_nets))
      if self._critical_patterns and isinstance(net_lengths_mm, dict):
        for name, net_len in net_lengths_mm.items():
          if any(pat.search(name) for pat in self._critical_patterns):
            critical_length_mm += float(net_len)
            critical_vias += float(net_vias.get(name, 0))
    else:
      completion = 0.0
      critical_length_mm = 0.0
      critical_vias = 0.0

    reward = (
        -length_mm * self._reward['length']
        -vias * self._reward['vias']
        -violations * self._reward['violations']
        -unconnected * self._reward['unconnected']
    )
    reward += self._reward['success'] if success else 0.0
    if failed:
      reward += self._reward['fail']
    if self._reward['runtime']:
      reward -= runtime * self._reward['runtime']
    if self._reward['completion']:
      reward += completion * self._reward['completion']
    if self._reward['critical_length']:
      reward -= critical_length_mm * self._reward['critical_length']
    if self._reward['critical_vias']:
      reward -= critical_vias * self._reward['critical_vias']
    for vtype, weight in self._reward.get('drc', {}).items():
      if not weight:
        continue
      if vtype == 'clearance':
        reward -= violations_clearance * weight
      elif vtype == 'hole_clearance':
        reward -= violations_hole_clearance * weight
      elif vtype == 'track_dangling':
        reward -= violations_track_dangling * weight
      elif vtype == 'via_dangling':
        reward -= violations_via_dangling * weight
    return dict(
        reward=np.float32(reward),
        length_mm=np.float32(length_mm),
        vias=np.float32(vias),
        violations=np.float32(violations),
        violations_clearance=np.float32(violations_clearance),
        violations_hole_clearance=np.float32(violations_hole_clearance),
        violations_track_dangling=np.float32(violations_track_dangling),
        violations_via_dangling=np.float32(violations_via_dangling),
        unconnected=np.float32(unconnected),
        completion=np.float32(completion),
        success=np.float32(success),
        failed=np.float32(failed),
        runtime=np.float32(runtime),
        critical_length_mm=np.float32(critical_length_mm),
        critical_vias=np.float32(critical_vias),
    )
