import math
from typing import Dict

import elements
import embodied
import numpy as np
import torch
import torch.nn as nn

from . import torch_rssm as rssm
from .torch_heads import MLPHead
from .torch_outs import Output
from .torch_utils import Normalize, SlowModel, LRSchedule


def sg(x: torch.Tensor, skip: bool = False) -> torch.Tensor:
  return x if skip else x.detach()


def sample(outputs: Dict[str, Output]) -> Dict[str, torch.Tensor]:
  return {k: v.sample() for k, v in outputs.items()}


def prefix(xs: Dict[str, torch.Tensor], p: str) -> Dict[str, torch.Tensor]:
  return {f'{p}/{k}': v for k, v in xs.items()}


def concat_dicts(dicts, dim: int):
  keys = dicts[0].keys()
  return {k: torch.cat([d[k] for d in dicts], dim=dim) for k in keys}


def isimage(space: elements.Space) -> bool:
  return space.dtype == np.uint8 and len(space.shape) == 3


class Agent(embodied.Agent):
  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    torch_cfg = getattr(config, 'torch', None)
    if isinstance(torch_cfg, dict):
      device = torch_cfg.get('device', 'auto')
    else:
      device = getattr(torch_cfg, 'device', 'auto')
    if device == 'auto':
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.device = torch.device(device)
    self.torch_cfg = torch_cfg

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}

    self.enc = rssm.Encoder(enc_space, **config.enc[config.enc.typ])
    self.dyn = rssm.RSSM(act_space, **config.dyn[config.dyn.typ])
    self.dec = rssm.Decoder(dec_space, **config.dec[config.dec.typ])

    self.feat2tensor = lambda x: torch.cat([
        x['deter'],
        x['stoch'].reshape(*x['stoch'].shape[:-2], -1)], dim=-1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = MLPHead(scalar, **config.rewhead)
    self.con = MLPHead(binary, **config.conhead)

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = MLPHead(act_space, outs, **config.policy)

    self.val = MLPHead(scalar, **config.value)
    self.slowval = SlowModel(
        MLPHead(scalar, **config.value),
        rate=config.slowvalue.rate,
        every=config.slowvalue.every,
    )

    self.retnorm = Normalize(**config.retnorm)
    self.valnorm = Normalize(**config.valnorm)
    self.advnorm = Normalize(**config.advnorm)

    self.model = nn.Module()
    self.model.enc = self.enc
    self.model.dyn = self.dyn
    self.model.dec = self.dec
    self.model.rew = self.rew
    self.model.con = self.con
    self.model.pol = self.pol
    self.model.val = self.val
    self.model.retnorm = self.retnorm
    self.model.valnorm = self.valnorm
    self.model.advnorm = self.advnorm
    self.model.to(self.device)
    self.slowval.to(self.device)

    self.opt_cfg = config.opt
    self._lr_schedule = LRSchedule(
        self.opt_cfg.lr,
        schedule=self.opt_cfg.schedule,
        warmup=self.opt_cfg.warmup,
        anneal=self.opt_cfg.anneal,
    )
    self.optimizer = torch.optim.Adam(
        self.model.parameters(),
        lr=self.opt_cfg.lr,
        betas=(self.opt_cfg.beta1, self.opt_cfg.beta2),
        eps=self.opt_cfg.eps,
        weight_decay=self.opt_cfg.wd,
    )

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    self.scales = scales

    self._updates = 0

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    def zeros(space):
      if np.issubdtype(space.dtype, np.floating):
        dtype = torch.float32
      elif np.issubdtype(space.dtype, np.integer):
        dtype = torch.int64
      elif space.dtype == bool:
        dtype = torch.bool
      else:
        dtype = torch.float32
      return torch.zeros((batch_size, *space.shape), device=self.device, dtype=dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        {k: zeros(v) for k, v in self.act_space.items()})

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    self.model.eval()
    with torch.no_grad():
      (enc_carry, dyn_carry, dec_carry, prevact) = carry
      obs_t = {k: torch.as_tensor(v, device=self.device) for k, v in obs.items()}
      reset = obs_t['is_first'].bool()
      enc_carry, enc_entry, tokens = self.enc(enc_carry, obs_t, reset, training=False, single=True)
      dyn_carry, dyn_entry, feat = self.dyn.observe(
          dyn_carry, tokens, prevact, reset, training=False, single=True)
      if dec_carry:
        dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, training=False, single=True)
      policy = self.pol(self.feat2tensor(feat), bdims=1)
      act = sample(policy)
      out = {}
      carry = (enc_carry, dyn_carry, dec_carry, act)
      act_np = {k: v.detach().cpu().numpy() for k, v in act.items()}
      return carry, act_np, out

  def train(self, carry, data):
    self.model.train()
    carry = self._detach_tree(carry)
    data_t = {k: torch.as_tensor(v, device=self.device) for k, v in data.items()}
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data_t)
    loss, (carry, entries, outs, metrics) = self.loss(carry, obs, prevact, training=True)

    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if isinstance(self.torch_cfg, dict):
      grad_clip = self.torch_cfg.get('grad_clip', 100.0)
    else:
      grad_clip = getattr(self.torch_cfg, 'grad_clip', 100.0) if self.torch_cfg is not None else 100.0
    if grad_clip and grad_clip > 0:
      nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
    lr = self._lr_schedule(self._updates)
    for group in self.optimizer.param_groups:
      group['lr'] = lr
    self.optimizer.step()
    self._updates += 1
    if self._updates > 0:
      self.slowval.update()

    outs_dict = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid,
          enc=entries[0],
          dyn=entries[1],
          dec=entries[2]))
      B, T = obs['is_first'].shape
      outs_dict['replay'] = {k: v.detach().cpu().numpy() for k, v in updates.items()}

    # Append last action to carry for the next rollout step.
    carry = (*carry, {k: data_t[k][:, -1] for k in self.act_space})

    metrics_np = {k: float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()}
    return carry, outs_dict, metrics_np

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first'].bool()
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = (~obs['is_terminal'].bool()).float()
    if self.config.contdisc:
      con = con * (1 - 1 / self.config.horizon)
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      target = value.float() / 255.0 if isimage(space) else value.float()
      losses[key] = recon.loss(target)

    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = {k: v[:, -K:].reshape(B * K, 1, *v.shape[2:]) for k, v in repfeat.items()}
    imgfeat = concat_dicts([
        {k: sg(v, skip=self.config.ac_grads) for k, v in first.items()},
        {k: sg(v, skip=self.config.ac_grads) for k, v in imgfeat.items()},
    ], 1)
    lastact = policyfn({k: v[:, -1] for k, v in imgfeat.items()})
    lastact = {k: v[:, None] for k, v in lastact.items()}
    imgact = concat_dicts([imgprevact, lastact], 1)
    inp = self.feat2tensor(imgfeat)
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp, 2).pred(),
        self.con(inp, 2).prob(1),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay value
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat = {k: v[:, -K:] for k, v in feat.items()}
      last, term, rew, boot = [x[:, -K:] for x in (last, term, rew, boot)]
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}
    self.model.eval()
    carry = self._detach_tree(carry)
    data_t = {k: torch.as_tensor(v, device=self.device) for k, v in data.items()}
    carry, obs, prevact, _ = self._apply_replay_context(carry, data_t)
    with torch.no_grad():
      _, (new_carry, entries, outs, mets) = self.loss(carry, obs, prevact, training=False)
    metrics = {k: float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v) for k, v in mets.items()}
    carry = (*new_carry, {k: data_t[k][:, -1] for k in self.act_space})
    return carry, metrics

  def stream(self, st):
    return st

  def save(self):
    return {
        'model': self.model.state_dict(),
        'slowval': self.slowval.state_dict(),
        'opt': self.optimizer.state_dict(),
        'updates': int(self._updates),
    }

  def load(self, data):
    self.model.load_state_dict(data['model'])
    self.slowval.load_state_dict(data['slowval'])
    self.optimizer.load_state_dict(data['opt'])
    self._updates = int(data.get('updates', 0))

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: torch.cat([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict({k: v.detach().cpu().numpy() for k, v in data.items()})
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    to_torch = lambda x: torch.as_tensor(x, device=self.device)
    rep_carry = (
        self.enc.truncate({k: to_torch(v[:, :K]) for k, v in entries[0].items()}, enc_carry),
        self.dyn.truncate({k: to_torch(v[:, :K]) for k, v in entries[1].items()}, dyn_carry),
        self.dec.truncate({k: to_torch(v[:, :K]) for k, v in entries[2].items()}, dec_carry),
    )
    obs_trim = {k: obs[k][:, K:] for k in obs}
    prevact_trim = {k: prevact[k][:, K:] for k in prevact}
    stepid_trim = stepid[:, K:]
    rep_obs = obs_trim
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = stepid_trim

    first_chunk = (data['consec'][:, 0] == 0)

    def _select(normal, replay):
      mask = first_chunk[:, None]
      while mask.ndim < normal.ndim:
        mask = mask.unsqueeze(-1)
      return torch.where(mask, replay, normal)

    def _select_tree(normal, replay):
      if isinstance(normal, dict):
        if not replay:
          return normal
        return {k: _select_tree(normal[k], replay.get(k, normal[k])) for k in normal}
      return _select(normal, replay)

    carry = tuple(_select_tree(n, r) for n, r in zip(carry, rep_carry))
    obs = {k: _select(obs_trim[k], rep_obs[k]) for k in obs_trim}
    prevact = {k: _select(prevact_trim[k], rep_prevact[k]) for k in prevact_trim}
    stepid = _select(stepid_trim, rep_stepid)
    return carry, obs, prevact, stepid

  def _detach_tree(self, xs):
    if torch.is_tensor(xs):
      return xs.detach()
    if isinstance(xs, dict):
      return {k: self._detach_tree(v) for k, v in xs.items()}
    if isinstance(xs, (list, tuple)):
      return type(xs)(self._detach_tree(v) for v in xs)
    return xs


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1.0 if contdisc else 1.0 - 1.0 / horizon
  weight = torch.cumprod(disc * con, dim=1) / disc
  last = torch.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(act[k])[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = torch.cat([tar_normed, torch.zeros_like(tar_normed[:, -1:])], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = adv.abs().mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (ret_normed.abs() >= 1.0).float().mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
  outs = {'ret': ret}
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1.0 - 1.0 / horizon
  weight = (~last.bool()).float()
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = torch.cat([ret_normed, torch.zeros_like(ret_normed[:, -1:])], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {'ret': ret}
  metrics = {}
  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  rets = [boot[:, -1]]
  live = (1 - term.float())[:, 1:] * disc
  cont = (1 - last.float())[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  rets = list(reversed(rets))
  return torch.stack(rets[:-1], dim=1)
