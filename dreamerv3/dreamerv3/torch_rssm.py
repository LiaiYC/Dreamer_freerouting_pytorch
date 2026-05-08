import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import elements

from .torch_nets import DictConcat, get_act, MLP
from .torch_utils import norm_layer, symlog
from . import torch_outs as outs


def _onehot_sample(logits: torch.Tensor, unimix: float = 0.0) -> torch.Tensor:
  dist = outs.OneHot(logits, unimix)
  return dist.sample()


class RSSM(nn.Module):
  def __init__(
      self,
      act_space: Dict[str, elements.Space],
      deter: int = 4096,
      hidden: int = 1024,
      stoch: int = 32,
      classes: int = 32,
      norm: str = 'rms',
      act: str = 'silu',
      unimix: float = 0.01,
      imglayers: int = 2,
      obslayers: int = 1,
      absolute: bool = False,
      free_nats: float = 1.0,
      **kwargs,
  ):
    super().__init__()
    self.act_space = act_space
    self.deter = int(deter)
    self.hidden = int(hidden)
    self.stoch = int(stoch)
    self.classes = int(classes)
    self.norm = norm
    self.act = get_act(act)
    self.unimix = float(unimix)
    self.imglayers = int(imglayers)
    self.obslayers = int(obslayers)
    self.absolute = bool(absolute)
    self.free_nats = float(free_nats)

    self._act_concat = DictConcat(act_space, fdims=1)
    act_dim = self._act_concat.dim
    stoch_dim = self.stoch * self.classes

    self.inp = nn.Linear(stoch_dim + act_dim, self.hidden)
    self.inp_norm = norm_layer(self.norm, self.hidden)
    self.gru = nn.GRUCell(self.hidden, self.deter)

    self.obs_mlp = nn.ModuleList([
        nn.Sequential(
            nn.LazyLinear(self.hidden),
            norm_layer(self.norm, self.hidden),
        )
        for _ in range(self.obslayers)
    ])
    self.obs_out = nn.Linear(self.hidden, stoch_dim)

    self.prior_mlp = nn.ModuleList([
        nn.Sequential(
            nn.LazyLinear(self.hidden),
            norm_layer(self.norm, self.hidden),
        )
        for _ in range(self.imglayers)
    ])
    self.prior_out = nn.Linear(self.hidden, stoch_dim)

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, (self.deter,)),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)),
    )

  def initial(self, batch_size: int) -> Dict[str, torch.Tensor]:
    device = next(self.parameters()).device
    return dict(
        deter=torch.zeros((batch_size, self.deter), device=device),
        stoch=torch.zeros((batch_size, self.stoch, self.classes), device=device),
    )

  def truncate(self, entries: Dict[str, torch.Tensor], carry=None):
    return {k: v[:, -1] for k, v in entries.items()}

  def starts(self, entries: Dict[str, torch.Tensor], carry, nlast: int):
    B = entries['deter'].shape[0]
    return {k: v[:, -nlast:].reshape(B * nlast, *v.shape[2:]) for k, v in entries.items()}

  def observe(self, carry, tokens, action, reset, training, single=False):
    if single:
      carry, (entry, feat) = self._observe(carry, tokens, action, reset, training)
      return carry, entry, feat
    B, T = reset.shape
    entries = {k: [] for k in carry}
    feats = {k: [] for k in ('deter', 'stoch', 'logit')}
    for t in range(T):
      carry, (entry, feat) = self._observe(
          carry,
          tokens[:, t],
          {k: v[:, t] for k, v in action.items()},
          reset[:, t],
          training,
      )
      for k in entries:
        entries[k].append(entry[k])
      for k in feats:
        feats[k].append(feat[k])
    entries = {k: torch.stack(v, dim=1) for k, v in entries.items()}
    feats = {k: torch.stack(v, dim=1) for k, v in feats.items()}
    return carry, entries, feats

  def _observe(self, carry, tokens, action, reset, training):
    reset = reset.bool()
    deter = carry['deter'] * (~reset).float().unsqueeze(-1)
    stoch = carry['stoch'] * (~reset).float().unsqueeze(-1).unsqueeze(-1)
    act = self._act_concat(action)
    act = act * (~reset).float().unsqueeze(-1)

    stoch_flat = stoch.reshape(stoch.shape[0], -1)
    inp = torch.cat([stoch_flat, act], dim=-1)
    inp = self.act(self.inp_norm(self.inp(inp)))
    deter = self.gru(inp, deter)

    token = tokens.reshape(tokens.shape[0], -1)
    if self.absolute:
      x = token
    else:
      x = torch.cat([deter, token], dim=-1)
    for layer in self.obs_mlp:
      x = self.act(layer(x))
    logit = self.obs_out(x)
    logit = logit.reshape(-1, self.stoch, self.classes)
    stoch = _onehot_sample(logit, self.unimix)

    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(carry) if callable(policy) else policy
      act = self._act_concat(action)
      stoch_flat = carry['stoch'].reshape(carry['stoch'].shape[0], -1)
      inp = torch.cat([stoch_flat, act], dim=-1)
      inp = self.act(self.inp_norm(self.inp(inp)))
      deter = self.gru(inp, carry['deter'])
      logit = self._prior(deter)
      stoch = _onehot_sample(logit, self.unimix)
      carry = dict(deter=deter, stoch=stoch)
      feat = dict(deter=deter, stoch=stoch, logit=logit)
      return carry, (feat, action)

    feats = {k: [] for k in ('deter', 'stoch', 'logit')}
    actions = {k: [] for k in self.act_space}
    for _ in range(length):
      carry, (feat, action) = self.imagine(carry, policy, 1, training, single=True)
      for k in feats:
        feats[k].append(feat[k])
      for k in actions:
        actions[k].append(action[k])
    feats = {k: torch.stack(v, dim=1) for k, v in feats.items()}
    actions = {k: torch.stack(v, dim=1) for k, v in actions.items()}
    return carry, feats, actions

  def loss(self, carry, tokens, acts, reset, training):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
    prior = self._prior(feat['deter'])
    post = feat['logit']
    prior_dist = self._dist(prior)
    post_dist = self._dist(post)
    dyn = self._dist(post.detach()).kl(prior_dist)
    rep = post_dist.kl(self._dist(prior.detach()))
    if self.free_nats:
      dyn = torch.maximum(dyn, torch.tensor(self.free_nats, device=dyn.device))
      rep = torch.maximum(rep, torch.tensor(self.free_nats, device=rep.device))
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = prior_dist.entropy().mean()
    metrics['rep_ent'] = post_dist.entropy().mean()
    return carry, entries, losses, feat, metrics

  def _prior(self, deter: torch.Tensor) -> torch.Tensor:
    x = deter
    for layer in self.prior_mlp:
      x = self.act(layer(x))
    logit = self.prior_out(x)
    return logit.reshape(*deter.shape[:-1], self.stoch, self.classes)

  def _dist(self, logits: torch.Tensor) -> outs.Output:
    dist = outs.OneHot(logits, self.unimix)
    return outs.Agg(dist, 1, torch.sum)


class Encoder(nn.Module):
  def __init__(self, obs_space, units=1024, norm='rms', act='silu', layers=3, use_symlog=True, **kwargs):
    super().__init__()
    if 'symlog' in kwargs:
      use_symlog = kwargs.pop('symlog')
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    if self.imgkeys:
      raise NotImplementedError('Image observations are not supported in torch encoder yet.')
    self.symlog = bool(use_symlog)
    self.concat = DictConcat(
        {k: obs_space[k] for k in self.veckeys},
        fdims=1,
        squish=symlog if self.symlog else None,
    )
    self.mlp = MLP(layers, units, act=act, norm=norm)

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size: int):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def forward(self, carry, obs, reset, training, single=False):
    bdims = 1 if single else 2
    if self.veckeys:
      vecs = {k: obs[k] for k in self.veckeys}
      x = self.concat(vecs)
      x = x.reshape(-1, x.shape[-1])
      x = self.mlp(x)
      x = x.reshape(*reset.shape, -1)
    else:
      x = torch.zeros((*reset.shape, 0), device=reset.device)
    entries = {}
    return carry, entries, x


class Decoder(nn.Module):
  def __init__(self, obs_space, units=1024, norm='rms', act='silu', layers=3, use_symlog=True, outscale=1.0, **kwargs):
    super().__init__()
    if 'symlog' in kwargs:
      use_symlog = kwargs.pop('symlog')
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    if self.imgkeys:
      raise NotImplementedError('Image observations are not supported in torch decoder yet.')
    self.symlog = bool(use_symlog)
    self.mlp = MLP(layers, units, act=act, norm=norm)
    spaces = {k: obs_space[k] for k in self.veckeys}
    outputs = {k: ('categorical' if v.discrete else ('symlog_mse' if self.symlog else 'mse')) for k, v in spaces.items()}
    from .torch_heads import DictHead
    self.head = DictHead(spaces, outputs, outscale=outscale, act=act, norm=norm)

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size: int):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def forward(self, carry, feat, reset, training, single=False):
    bshape = reset.shape
    stoch = feat['stoch']
    deter = feat['deter']
    if stoch.ndim > 2:
      stoch = stoch.reshape(*stoch.shape[:-2], -1)
    inp = torch.cat([stoch, deter], dim=-1)
    inp = inp.reshape(-1, inp.shape[-1])
    x = self.mlp(inp)
    x = x.reshape(*bshape, -1)
    recons = self.head(x)
    entries = {}
    return carry, entries, recons
