import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter


def symlog(x: torch.Tensor) -> torch.Tensor:
  return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
  return torch.sign(x) * torch.expm1(torch.abs(x))


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-4, affine: bool = True):
    super().__init__()
    self.eps = eps
    self.affine = affine
    if affine:
      self.scale = nn.Parameter(torch.ones(dim))
    else:
      self.register_parameter('scale', None)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.dtype in (torch.float16, torch.bfloat16):
      x_float = x.float()
    else:
      x_float = x
    rms = torch.mean(x_float * x_float, dim=-1, keepdim=True)
    x_norm = x_float * torch.rsqrt(rms + self.eps)
    if self.affine:
      x_norm = x_norm * self.scale
    return x_norm.to(x.dtype)


def norm_layer(name: str, dim: int) -> nn.Module:
  if name == 'none':
    return nn.Identity()
  if name == 'rms':
    return RMSNorm(dim)
  if name == 'layer':
    return nn.LayerNorm(dim)
  raise NotImplementedError(name)


class Normalize(nn.Module):
  def __init__(
      self,
      impl: str = 'none',
      rate: float = 0.01,
      limit: float = 1e-8,
      perclo: float = 5.0,
      perchi: float = 95.0,
      debias: bool = True,
  ):
    super().__init__()
    self.impl = impl
    self.rate = float(rate)
    self.limit = float(limit)
    self.perclo = float(perclo)
    self.perchi = float(perchi)
    self.debias = bool(debias)
    self.register_buffer('device_ref', torch.zeros(()))
    if self.impl == 'none':
      pass
    elif self.impl == 'meanstd':
      self.register_buffer('mean', torch.zeros(()))
      self.register_buffer('sqrs', torch.zeros(()))
    elif self.impl == 'perc':
      self.register_buffer('lo', torch.zeros(()))
      self.register_buffer('hi', torch.zeros(()))
    else:
      raise NotImplementedError(self.impl)
    if self.debias and self.impl != 'none':
      self.register_buffer('corr', torch.zeros(()))

  def forward(self, x: torch.Tensor, update: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if update:
      self.update(x)
    return self.stats()

  def update(self, x: torch.Tensor) -> None:
    x = x.detach().float()
    if self.impl == 'none':
      pass
    elif self.impl == 'meanstd':
      self._update('mean', x.mean())
      self._update('sqrs', (x * x).mean())
    elif self.impl == 'perc':
      flat = x.flatten()
      lo = torch.quantile(flat, self.perclo / 100.0)
      hi = torch.quantile(flat, self.perchi / 100.0)
      self._update('lo', lo)
      self._update('hi', hi)
    else:
      raise NotImplementedError(self.impl)
    if self.debias and self.impl != 'none':
      self._update('corr', torch.tensor(1.0, device=x.device))

  def stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.impl == 'none':
      return torch.tensor(0.0, device=self._device()), torch.tensor(1.0, device=self._device())
    corr = torch.tensor(1.0, device=self._device())
    if self.debias and self.impl != 'none':
      corr = corr / torch.maximum(torch.tensor(self.rate, device=self._device()), self.corr)
    if self.impl == 'meanstd':
      mean = self.mean * corr
      std = torch.sqrt(torch.clamp(self.sqrs * corr - mean * mean, min=0.0))
      std = torch.maximum(std, torch.tensor(self.limit, device=self._device()))
      return mean, std
    if self.impl == 'perc':
      lo = self.lo * corr
      hi = self.hi * corr
      scale = torch.maximum(torch.tensor(self.limit, device=self._device()), hi - lo)
      return lo.detach(), scale.detach()
    raise NotImplementedError(self.impl)

  def _update(self, name: str, value: torch.Tensor) -> None:
    buf = getattr(self, name)
    updated = (1 - self.rate) * buf + self.rate * value.detach()
    buf.copy_(updated)

  def _device(self) -> torch.device:
    return self.device_ref.device


class SlowModel(nn.Module):
  def __init__(self, model: nn.Module, rate: float = 0.02, every: int = 1):
    super().__init__()
    self.source = model
    self.model = copy.deepcopy(model)
    for param in self.model.parameters():
      param.requires_grad = False
    self.rate = float(rate)
    self.every = int(every)
    self.register_buffer('count', torch.zeros((), dtype=torch.int64))

  def forward(self, *args, **kwargs):
    return self.model(*args, **kwargs)

  def update(self):
    if self._has_uninitialized(self.source):
      return
    if self._has_uninitialized(self.model):
      self._init_from_source()
    if self.count.item() % self.every == 0:
      with torch.no_grad():
        for src, dst in zip(self.source.parameters(), self.model.parameters()):
          dst.data.mul_(1.0 - self.rate).add_(src.data, alpha=self.rate)
    self.count += 1

  def _has_uninitialized(self, module: nn.Module) -> bool:
    for param in module.parameters():
      if isinstance(param, UninitializedParameter):
        return True
    return False

  def _init_from_source(self):
    self.model = copy.deepcopy(self.source)
    for param in self.model.parameters():
      param.requires_grad = False
    # Ensure device matches source parameters.
    try:
      device = next(self.source.parameters()).device
      self.model.to(device)
    except StopIteration:
      pass


class LRSchedule:
  def __init__(self, lr: float, schedule: str = 'const', warmup: int = 0, anneal: int = 0):
    self.lr = float(lr)
    self.schedule = schedule
    self.warmup = int(warmup)
    self.anneal = int(anneal)

  def __call__(self, step: int) -> float:
    step = int(step)
    if self.warmup > 0 and step < self.warmup:
      return self.lr * (step + 1) / self.warmup
    if self.schedule == 'const' or self.anneal <= 0:
      return self.lr
    progress = min(max(step - self.warmup, 0), self.anneal)
    frac = progress / max(self.anneal, 1)
    if self.schedule == 'linear':
      return self.lr * (1.0 - 0.9 * frac)
    if self.schedule == 'cosine':
      return self.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
    raise NotImplementedError(self.schedule)
