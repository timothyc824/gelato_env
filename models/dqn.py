# WIP

from copy import deepcopy
from typing import Optional, List

import pytorch_lightning as pl
import torch
from torch.nn import Module

from models.mc_control import Trajectory


class DQN(pl.LightningModule):

    def __init__(self,
                 env: GelateriaEnv,
                 actor_net: Module,
                 name: str = "dqn",
                 lr: float = 1e-3,
                 gamma: float = 0.9,
                 epsilon: float = 0.9,
                 warm_start: Optional[int] = None,
                 critic_net: Optional[Module] = None,
                 ):
        super().__init__()
        self._env = env
        self._lr = lr
        self._gamma = gamma
        self._epsilon = epsilon
        self._warm_start = warm_start
        self._name = name
        self._actor = actor_net
        self._critic = critic_net if critic_net is not None else actor_net
        self._input_space = self._get_input_space()

    def _get_input_space(self):
        n_flavours = self._env.state_space_size[0]
        return torch.cartesian_prod(*torch.arange(0, 101, 1).tile(2, 1)).tile(n_flavours, 1).reshape(n_flavours, -1, 2)

    def _Q(self, input: torch.Tensor):
        return self._actor(input)

    @property
    def q_values(self) -> torch.Tensor:
        return self._Q(self._input_space).squeeze(axis=0)

    @property
    def q_values_mean_normalised(self):
        """
        Returns the Q values with the mean subtracted from each row.
        """
        q_values = self.q_values
        means = q_values.mean(dim=-1)
        return (q_values - torch.concat(means)).squeeze(axis=0)

    def _select_action(self, current_stock: List[int]):
        """
        Selects an action from the action space.
        Args:
            current_stock: The current stock of each flavour.

        Returns:
            The action to take.
        """
        if self._rng.random() <= self._epsilon:
            return self._select_greedy_action(current_stock=current_stock)
        else:
            return self._select_random_action()

    def _select_greedy_action(self, current_stock: List[int]):
        """
        Selects the greedy action from the action space.

        Args:
            current_stock: The current stock of each flavour.

        Returns:
            The greedy actions to take.
        """
        n_flavours = len(current_stock)
        idxs = torch.concat([torch.arange(101*st, 101*(st+1)) for st in current_stock]).reshape(n_flavours, -1)
        reduced_space = torch.concat([self._input_space[i, idxs[i], :] for i in range(n_flavours)])
        return torch.argmax(self._Q(reduced_space), dim=-1) / 100

    def _select_random_action(self):
        """Selects a random action from the action space with uniform probability."""
        return self._rng.integers(low=0, high=101, size=self._dims[0]) / 100

    def _train_step(self, trajectory: Trajectory):
        ...



