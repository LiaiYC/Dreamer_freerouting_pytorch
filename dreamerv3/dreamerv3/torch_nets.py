import math
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import elements

from .torch_utils import RMSNorm, norm_layer, symlog


def get_act(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
  if name == 'none':
    return lambda x: x
  if name == 'mish':
    return lambda x: x * torch.tanh(F.softplus(x))
  if name == 'relu2':
    return lambda x: F.relu(x) ** 2
  if name == 'swiglu':
    def fn(x):
      x, y = torch.chunk(x, 2, dim=-1)
      return F.silu(x) * y
    return fn
  if hasattr(F, name):
    return getattr(F, name)
  if name == 'silu':
    return F.silu
  if name == 'gelu':
    return F.gelu
  raise NotImplementedError(name)


class Linear(nn.Module):
  def __init__(self, out_dim: int, in_dim: int | None = None, bias: bool = True, outscale: float = 1.0):
    super().__init__()
    if in_dim is None:
      self.linear = nn.LazyLinear(out_dim, bias=bias)
    else:
      self.linear = nn.Linear(in_dim, out_dim, bias=bias)
    self.outscale = float(outscale)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = self.linear(x)
    if self.outscale != 1.0:
      y = y * self.outscale
    return y


class MLP(nn.Module):
  def __init__(self, layers: int = 5, units: int = 1024, act: str = 'silu', norm: str = 'rms'):
    super().__init__()
    self.layers = int(layers)
    self.units = int(units)
    self.act = get_act(act)
    self.linears = nn.ModuleList([Linear(self.units) for _ in range(self.layers)])
    self.norms = nn.ModuleList([norm_layer(norm, self.units) for _ in range(self.layers)])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    for linear, norm in zip(self.linears, self.norms):
      x = linear(x)
      x = norm(x)
      x = self.act(x)
    x = x.reshape(*shape, x.shape[-1])
    return x


class DictConcat:
  def __init__(self, spaces: Dict[str, elements.Space], fdims: int, squish: Callable[[torch.Tensor], torch.Tensor] | None = None):
    assert 1 <= fdims
    self.keys = sorted(spaces.keys())
    self.spaces = spaces
    self.fdims = fdims
    self.squish = squish or (lambda x: x)
    self.dim = self._compute_dim()

  def _compute_dim(self) -> int:
    total = 0
    for key in self.keys:
      space = self.spaces[key]
      if space.discrete:
        classes = np.asarray(space.classes).flatten()
        assert (classes == classes[0]).all(), classes
        total += int(np.prod(space.shape)) * int(classes[0])
      else:
        total += int(np.prod(space.shape))
    return total

  def __call__(self, xs: Dict[str, torch.Tensor]) -> torch.Tensor:
    ys = []
    for key in self.keys:
      space = self.spaces[key]
      x = xs[key]
      if space.discrete:
        classes = np.asarray(space.classes).flatten()
        assert (classes == classes[0]).all(), classes
        classes = int(classes[0])
        x = x.long()
        x = F.one_hot(x, classes).float()
      else:
        x = self.squish(x.float())
      bdims = x.ndim - len(space.shape)
      x = x.reshape(*x.shape[:bdims + self.fdims - 1], -1)
      ys.append(x)
    return torch.cat(ys, dim=-1)
