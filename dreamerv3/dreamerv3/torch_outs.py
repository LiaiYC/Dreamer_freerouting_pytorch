import math
from typing import Iterable

import torch
import torch.nn.functional as F


class Output:
  def pred(self):
    raise NotImplementedError

  def loss(self, target: torch.Tensor) -> torch.Tensor:
    return -self.logp(target.detach())

  def sample(self) -> torch.Tensor:
    raise NotImplementedError

  def logp(self, event: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

  def prob(self, event: torch.Tensor) -> torch.Tensor:
    return torch.exp(self.logp(event))

  def entropy(self) -> torch.Tensor:
    raise NotImplementedError

  def kl(self, other: 'Output') -> torch.Tensor:
    raise NotImplementedError


class Agg(Output):
  def __init__(self, output: Output, dims: int, agg=torch.sum):
    self.output = output
    self.axes = tuple(range(-dims, 0))
    self.agg = agg

  def pred(self):
    return self.output.pred()

  def loss(self, target: torch.Tensor) -> torch.Tensor:
    loss = self.output.loss(target)
    return self.agg(loss, dim=self.axes)

  def sample(self) -> torch.Tensor:
    return self.output.sample()

  def logp(self, event: torch.Tensor) -> torch.Tensor:
    return self.output.logp(event).sum(self.axes)

  def prob(self, event: torch.Tensor) -> torch.Tensor:
    return self.output.prob(event).sum(self.axes)

  def entropy(self) -> torch.Tensor:
    ent = self.output.entropy()
    return self.agg(ent, dim=self.axes)

  def kl(self, other: 'Agg') -> torch.Tensor:
    return self.agg(self.output.kl(other.output), dim=self.axes)


class MSE(Output):
  def __init__(self, mean: torch.Tensor, squash=None):
    self.mean = mean.float()
    self.squash = squash or (lambda x: x)

  def pred(self):
    return self.mean

  def loss(self, target: torch.Tensor) -> torch.Tensor:
    target = self.squash(target.float()).detach()
    return (self.mean - target) ** 2


class Huber(Output):
  def __init__(self, mean: torch.Tensor, eps: float = 1.0):
    self.mean = mean.float()
    self.eps = float(eps)

  def pred(self):
    return self.mean

  def loss(self, target: torch.Tensor) -> torch.Tensor:
    target = target.float().detach()
    dist = self.mean - target
    return torch.sqrt(dist * dist + self.eps * self.eps) - self.eps


class Normal(Output):
  def __init__(self, mean: torch.Tensor, stddev: torch.Tensor | float = 1.0):
    self.mean = mean.float()
    if isinstance(stddev, torch.Tensor):
      self.stddev = stddev.float()
    else:
      self.stddev = torch.full_like(self.mean, float(stddev))

  def pred(self):
    return self.mean

  def sample(self) -> torch.Tensor:
    return self.mean + self.stddev * torch.randn_like(self.mean)

  def logp(self, event: torch.Tensor) -> torch.Tensor:
    event = event.float()
    var = self.stddev ** 2
    return -0.5 * ((event - self.mean) ** 2 / var + torch.log(2 * math.pi * var))

  def entropy(self) -> torch.Tensor:
    return 0.5 * torch.log(2 * math.pi * (self.stddev ** 2)) + 0.5

  def kl(self, other: 'Normal') -> torch.Tensor:
    return 0.5 * (
        (self.stddev / other.stddev) ** 2 +
        ((other.mean - self.mean) / other.stddev) ** 2 +
        2 * torch.log(other.stddev) - 2 * torch.log(self.stddev) - 1
    )


class Binary(Output):
  def __init__(self, logit: torch.Tensor):
    self.logit = logit.float()

  def pred(self):
    return (self.logit > 0).float()

  def logp(self, event: torch.Tensor) -> torch.Tensor:
    event = torch.as_tensor(event, device=self.logit.device, dtype=self.logit.dtype)
    logp = F.logsigmoid(self.logit)
    lognotp = F.logsigmoid(-self.logit)
    return event * logp + (1 - event) * lognotp

  def sample(self) -> torch.Tensor:
    prob = torch.sigmoid(self.logit)
    return torch.bernoulli(prob)


class Categorical(Output):
  def __init__(self, logits: torch.Tensor, unimix: float = 0.0):
    logits = logits.float()
    if unimix:
      probs = F.softmax(logits, dim=-1)
      uniform = torch.ones_like(probs) / probs.shape[-1]
      probs = (1 - unimix) * probs + unimix * uniform
      logits = torch.log(probs)
    self.logits = logits

  def pred(self):
    return torch.argmax(self.logits, dim=-1)

  def sample(self) -> torch.Tensor:
    dist = torch.distributions.Categorical(logits=self.logits)
    return dist.sample()

  def logp(self, event: torch.Tensor) -> torch.Tensor:
    event = torch.as_tensor(event, device=self.logits.device).long()
    onehot = F.one_hot(event, self.logits.shape[-1]).float()
    return (F.log_softmax(self.logits, dim=-1) * onehot).sum(dim=-1)

  def entropy(self) -> torch.Tensor:
    logprob = F.log_softmax(self.logits, dim=-1)
    prob = F.softmax(self.logits, dim=-1)
    return -(prob * logprob).sum(dim=-1)

  def kl(self, other: 'Categorical') -> torch.Tensor:
    logprob = F.log_softmax(self.logits, dim=-1)
    logother = F.log_softmax(other.logits, dim=-1)
    prob = F.softmax(self.logits, dim=-1)
    return (prob * (logprob - logother)).sum(dim=-1)


class OneHot(Output):
  def __init__(self, logits: torch.Tensor, unimix: float = 0.0):
    self.dist = Categorical(logits, unimix)

  def pred(self):
    index = self.dist.pred()
    return self._onehot_with_grad(index)

  def sample(self) -> torch.Tensor:
    index = self.dist.sample()
    return self._onehot_with_grad(index)

  def logp(self, event: torch.Tensor) -> torch.Tensor:
    return (F.log_softmax(self.dist.logits, dim=-1) * event).sum(dim=-1)

  def entropy(self) -> torch.Tensor:
    return self.dist.entropy()

  def kl(self, other: 'OneHot') -> torch.Tensor:
    return self.dist.kl(other.dist)

  def _onehot_with_grad(self, index: torch.Tensor) -> torch.Tensor:
    value = F.one_hot(index, self.dist.logits.shape[-1]).float()
    probs = F.softmax(self.dist.logits, dim=-1)
    return value.detach() + (probs - probs.detach())


class TwoHot(Output):
  def __init__(self, logits: torch.Tensor, bins: torch.Tensor, squash=None, unsquash=None):
    logits = logits.float()
    self.logits = logits
    self.probs = F.softmax(logits, dim=-1)
    self.bins = bins.float()
    self.squash = squash or (lambda x: x)
    self.unsquash = unsquash or (lambda x: x)

  def pred(self):
    n = self.logits.shape[-1]
    if n % 2 == 1:
      m = (n - 1) // 2
      p1 = self.probs[..., :m]
      p2 = self.probs[..., m:m + 1]
      p3 = self.probs[..., m + 1:]
      b1 = self.bins[..., :m]
      b2 = self.bins[..., m:m + 1]
      b3 = self.bins[..., m + 1:]
      wavg = (p2 * b2).sum(dim=-1) + ((p1 * b1).flip(-1) + (p3 * b3)).sum(dim=-1)
      return self.unsquash(wavg)
    p1 = self.probs[..., :n // 2]
    p2 = self.probs[..., n // 2:]
    b1 = self.bins[..., :n // 2]
    b2 = self.bins[..., n // 2:]
    wavg = ((p1 * b1).flip(-1) + (p2 * b2)).sum(dim=-1)
    return self.unsquash(wavg)

  def loss(self, target: torch.Tensor) -> torch.Tensor:
    target = self.squash(target.float()).detach()
    bins = self.bins
    below = (bins <= target[..., None]).long().sum(dim=-1) - 1
    above = len(bins) - (bins > target[..., None]).long().sum(dim=-1)
    below = torch.clamp(below, 0, len(bins) - 1)
    above = torch.clamp(above, 0, len(bins) - 1)
    equal = below == above
    dist_to_below = torch.where(equal, torch.ones_like(target), torch.abs(bins[below] - target))
    dist_to_above = torch.where(equal, torch.ones_like(target), torch.abs(bins[above] - target))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target_dist = (
        F.one_hot(below, len(bins)).float() * weight_below[..., None] +
        F.one_hot(above, len(bins)).float() * weight_above[..., None]
    )
    log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
    return -(target_dist * log_pred).sum(dim=-1)
