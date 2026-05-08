from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import elements

from . import torch_outs as outs
from .torch_nets import Linear, MLP
from .torch_utils import symlog, symexp


class MLPHead(nn.Module):
  def __init__(self, space, output, layers=5, units=1024, act='silu', norm='rms', **hkw):
    super().__init__()
    self.mlp = MLP(layers, units, act=act, norm=norm)
    allowed = {'minstd', 'maxstd', 'unimix', 'bins', 'outscale'}
    head_kw = {k: v for k, v in hkw.items() if k in allowed}
    if isinstance(space, dict):
      self.head = DictHead(space, output, **head_kw)
    else:
      self.head = Head(space, output, **head_kw)

  def forward(self, x: torch.Tensor, bdims: int):
    bshape = x.shape[:bdims]
    x = x.reshape(*bshape, -1)
    x = self.mlp(x)
    return self.head(x)


class DictHead(nn.Module):
  def __init__(self, spaces: Dict[str, elements.Space], outputs, **kw):
    super().__init__()
    if not isinstance(outputs, dict):
      outputs = {k: outputs for k in spaces}
    self.spaces = spaces
    self.outputs = outputs
    self.kw = kw
    self.heads = nn.ModuleDict({
        k: Head(spaces[k], outputs[k], **kw) for k in spaces
    })

  def forward(self, x: torch.Tensor):
    return {k: self.heads[k](x) for k in self.spaces}


class Head(nn.Module):
  def __init__(
      self,
      space,
      output,
      minstd: float = 0.1,
      maxstd: float = 1.0,
      unimix: float = 0.0,
      bins: int = 255,
      outscale: float = 1.0,
      **kwargs,
  ):
    super().__init__()
    if isinstance(space, tuple):
      space = elements.Space(np.float32, space)
    if output == 'onehot':
      classes = np.asarray(space.classes).flatten()
      assert (classes == classes[0]).all(), classes
      shape = (*space.shape, classes[0].item())
      space = elements.Space(np.float32, shape, 0.0, 1.0)
    self.space = space
    self.impl = output
    self.minstd = float(minstd)
    self.maxstd = float(maxstd)
    self.unimix = float(unimix)
    self.bins = int(bins)
    self.outscale = float(outscale)
    self._layers = nn.ModuleDict()

  def forward(self, x: torch.Tensor):
    if not hasattr(self, self.impl):
      raise NotImplementedError(self.impl)
    output = getattr(self, self.impl)(x)
    if self.space.shape:
      output = outs.Agg(output, len(self.space.shape), torch.sum)
    return output

  def _linear(self, name: str, x: torch.Tensor, shape, outscale=1.0):
    if isinstance(shape, int):
      out_dim = shape
      reshape = False
    else:
      out_dim = int(np.prod(shape))
      reshape = True
    if name not in self._layers:
      layer = Linear(out_dim, outscale=outscale)
      self._layers[name] = layer.to(x.device)
    y = self._layers[name](x)
    if reshape:
      y = y.reshape(*x.shape[:-1], *shape)
    return y

  def binary(self, x: torch.Tensor):
    logit = self._linear('binary', x, self.space.shape, outscale=self.outscale)
    return outs.Binary(logit)

  def categorical(self, x: torch.Tensor):
    assert self.space.discrete
    classes = np.asarray(self.space.classes).flatten()
    assert (classes == classes[0]).all(), classes
    shape = (*self.space.shape, classes[0].item())
    logits = self._linear('categorical', x, shape, outscale=self.outscale)
    output = outs.Categorical(logits)
    output.minent = 0.0
    output.maxent = float(np.log(logits.shape[-1]))
    return output

  def onehot(self, x: torch.Tensor):
    logits = self._linear('onehot', x, self.space.shape, outscale=self.outscale)
    return outs.OneHot(logits, self.unimix)

  def mse(self, x: torch.Tensor):
    pred = self._linear('mse', x, self.space.shape, outscale=self.outscale)
    return outs.MSE(pred)

  def huber(self, x: torch.Tensor):
    pred = self._linear('huber', x, self.space.shape, outscale=self.outscale)
    return outs.Huber(pred)

  def symlog_mse(self, x: torch.Tensor):
    pred = self._linear('symlog_mse', x, self.space.shape, outscale=self.outscale)
    return outs.MSE(pred, symlog)

  def symexp_twohot(self, x: torch.Tensor):
    shape = (*self.space.shape, self.bins)
    logits = self._linear('symexp_twohot', x, shape, outscale=self.outscale)
    device = logits.device
    if self.bins % 2 == 1:
      half = torch.linspace(-20, 0, (self.bins - 1) // 2 + 1, device=device)
      half = symexp(half)
      bins = torch.cat([half, -half[:-1].flip(0)], 0)
    else:
      half = torch.linspace(-20, 0, self.bins // 2, device=device)
      half = symexp(half)
      bins = torch.cat([half, -half.flip(0)], 0)
    return outs.TwoHot(logits, bins)

  def bounded_normal(self, x: torch.Tensor):
    mean = self._linear('bounded_normal_mean', x, self.space.shape, outscale=self.outscale)
    stddev = self._linear('bounded_normal_std', x, self.space.shape, outscale=self.outscale)
    lo, hi = self.minstd, self.maxstd
    stddev = (hi - lo) * torch.sigmoid(stddev + 2.0) + lo
    output = outs.Normal(torch.tanh(mean), stddev)
    output.minent = outs.Normal(torch.zeros_like(mean), torch.tensor(self.minstd, device=mean.device)).entropy()
    output.maxent = outs.Normal(torch.zeros_like(mean), torch.tensor(self.maxstd, device=mean.device)).entropy()
    return output

  def normal_logstd(self, x: torch.Tensor):
    mean = self._linear('normal_logstd_mean', x, self.space.shape, outscale=self.outscale)
    stddev = self._linear('normal_logstd_std', x, self.space.shape, outscale=self.outscale)
    return outs.Normal(mean, torch.exp(stddev))
