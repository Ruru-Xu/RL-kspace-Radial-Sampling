import numpy as np
import scipy.signal

import torch
import torch.nn as nn

from rl.nn_utils import MaskedCategorical
from rl.ppo_core_net_mt import Kspace_Net_MT, Kspace_Net_Critic_MT


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class KspaceMaskedCategoricalActor_MT(Actor):

    def __init__(self, act_dim, feature_dim, mt_shape):
        super().__init__()
        self.logits_net = Kspace_Net_MT(act_dim, feature_dim, mt_shape)

    def _distribution(self, obs, mask):
        logits = self.logits_net(obs)
        return MaskedCategorical(logits=logits, mask=mask)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class KspaceMaskedActorCritic_MT(nn.Module):

    def __init__(self, action_space, feature_dim=50, mt_shape=256):

        super().__init__()
        self._cur_mask = None
        self.pi = KspaceMaskedCategoricalActor_MT(action_space.n, feature_dim, mt_shape)
        self.v = Kspace_Net_Critic_MT(feature_dim, mt_shape)

    def get_action_and_value(self, obs, mask, a=None, deterministic=False):
        pi = self.pi._distribution(obs, mask)
        if a is None:
            if deterministic:

                a = pi.mode
            else:
                a = pi.sample()
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)

        dist_entropy = pi.entropy()

        return a, logp_a, dist_entropy, v

    def get_action_and_value_aux(self, obs, mask, a=None, deterministic=False):

        pi = self.pi._distribution(obs, mask)
        if a is None:
            if deterministic:
                a = pi.mode
            else:
                a = pi.sample()
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)
        dist_entropy = pi.entropy()

        return a, logp_a, dist_entropy, v

    def act(self, obs):
        return self.step(obs)[0]

    def get_value(self, obs):
        return self.v(obs)
