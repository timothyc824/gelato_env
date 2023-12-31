from typing import Optional, Tuple, Sequence
import torch
import torch.nn as nn
import numpy as np

from models.model import BaseNetwork
import torch.nn.functional as F
from utils.distributions import CategoricalMasked


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class ActorNetwork(BaseNetwork):
    def __init__(self, num_observation: int, num_action: int, hidden_layers: Optional[Sequence[int]] = (128, 128),
                 activation: Optional[str] = "ReLU"):
        """Initialize parameters and build actor network.

        Args:
            num_observation (int): Dimension of each state observation
            num_action (int): Dimension of each action
            hidden_layers (list): Number of nodes in hidden layers of the network (default: (32, 64))
            activation (str): Activation function to use between hidden layers (default: "ReLU")
        """
        super(ActorNetwork, self).__init__()

        in_dim = num_observation
        layers = []
        if hidden_layers is not None:
            for dim in hidden_layers:
                layers += [nn.Linear(in_features=in_dim, out_features=dim)]
                in_dim = dim
                if activation is not None:
                    layers += [getattr(nn, activation)()]
        layers += [nn.Linear(in_features=in_dim, out_features=num_action)]
        self._model = nn.Sequential(*layers)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        logits = self._model(state)
        return logits

    def evaluate(self, state_obs, epsilon: float = 1e-6, mask: Optional[np.ndarray] = None) \
            -> Tuple[Sequence[float], torch.Tensor, torch.Tensor]:

        logits = self.forward(state_obs.float())
        if mask is None:
            mask = np.ones(logits.shape, dtype=bool)
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).bool()
        else:
            mask_tensor = mask.bool()

        dist = CategoricalMasked(logits=logits, mask=mask_tensor)
        with torch.no_grad():
            action_numpy = np.array([None] * len(logits))
            action_numpy[mask_tensor.any(dim=-1).detach().cpu().numpy()] = (dist.sample().detach().cpu().int()).numpy()
            action = action_numpy
        action_probs = dist.probs
        log_action_probabilities = F.log_softmax(logits, dim=-1)
        return action, action_probs, log_action_probabilities

    def get_det_action(self, state_obs: torch.Tensor, mask: Optional[np.ndarray] = None) -> np.ndarray:
        with torch.no_grad():

            logits = self.forward(state_obs)

            if mask is None:
                mask = np.ones(logits.shape, dtype=bool)
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask).bool()
            else:
                mask_tensor = mask.bool()

            dist = CategoricalMasked(logits=logits, mask=mask_tensor)
            with torch.no_grad():
                det_action = torch.argmax(dist.probs, dim=1).detach().cpu().int().numpy()

            return det_action


class SoftQNetwork(BaseNetwork):
    """Critic (Value) Model."""

    def __init__(self, num_observation, num_action, hidden_layers: Optional[Sequence[int]] = (128, 128),
                 activation: Optional[str] = "ReLU", seed: int = 1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(SoftQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        in_dim = num_observation
        layers = []
        if hidden_layers is not None:
            for dim in hidden_layers:
                layers += [nn.Linear(in_features=in_dim, out_features=dim)]
                in_dim = dim
                if activation is not None:
                    layers += [getattr(nn, activation)()]
        self.last_layer = nn.Linear(in_features=in_dim, out_features=num_action)
        self.hidden_layers = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(*hidden_init(layer))
        self.last_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = state
        x = self.hidden_layers(x)
        return self.last_layer(x)
