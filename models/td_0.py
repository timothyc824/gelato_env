from copy import deepcopy
from dataclasses import dataclass
from typing import List, Union, Optional, Tuple
import logging
import tqdm
import numpy as np

from env.gelateria_env import GelateriaEnv

logger = logging.getLogger(__name__)


@dataclass
class StateQuad:
    stock: Union[List[int], np.array]
    actions: Union[List[float], np.array]
    reward: Union[List[float], np.array]
    next_stock: Union[List[int], np.array]

    def __post_init__(self):
        self.stock = np.array(self.stock)
        self.actions = np.round(100 * np.array(self.actions), 2).astype(int)
        self.reward = np.array(self.reward)
        self.next_stock = np.array(self.next_stock)

    def quad(self):
        return self.stock, self.actions, self.reward, self.next_stock


class TDZero:

    def __init__(self,
                 env: GelateriaEnv,
                 n_episodes: int,
                 horizon_steps: int,
                 name: str = "TD_0",
                 epsilon: float = 0.9,
                 gamma: float = 0.9,
                 alpha: float = 0.9,
                 warm_start: Optional[int] = None,
                 q_init: Optional[np.array] = None,
                 ):
        self._env = env
        self._n_episodes = n_episodes
        self._horizon_steps = horizon_steps
        self._name = name
        self._gamma = gamma
        self._epsilon = epsilon
        self._alpha = alpha
        self._warm_start = warm_start

        # dims: (n_flavours, stock, reductions)
        # usually taken to be (n_flavours, 101, 101)
        self._dims = env.state_space_size
        self._Q = q_init if q_init is not None else np.random.normal(size=self._dims)
        self._G = np.zeros(self._dims[0], dtype=np.float16)
        self._policy = np.zeros(self._dims[:-1], dtype=np.float16)

        self._rng = np.random.default_rng(seed=42)

        self._rewards = []
        self._discounted_rewards = []

    @property
    def policy(self):
        return self._policy.squeeze(axis=0)

    @property
    def q_values(self):
        return self._Q.squeeze(axis=0)

    @property
    def q_values_mean_normalised(self):
        means = []
        for idx in range(self._Q.shape[0]):
            means.append(np.mean(self._Q[idx]))
        return (self._Q - np.array(means)).squeeze(axis=0)

    def _select_action(self, current_stock: List[int], mask: np.array):
        """
        Selects an action from the masked action space.
        Args:
            current_stock: The current stock of each flavour.
            mask: Mask invalid actions.

        Returns:
            The action to take.
        """
        if self._rng.random() <= self._epsilon:
            return self._select_greedy_action(current_stock=current_stock, mask=mask)
        else:
            return self._select_random_action(mask=mask)

    def _select_greedy_action(self, current_stock: List[int], mask: np.array):
        """
        Selects the greedy action from the action space.

        Args:
            current_stock: The current stock of each flavour.
            mask: Masked actions in the current state. By default, lower reductions are masked out.

        Returns:
            The greedy actions to take.
        """
        actions = (np.argmax(self._Q + mask, axis=-1) / 100)[:, current_stock]
        return actions

    def _select_random_action(self, mask: np.array):
        """Selects a random action from the masked action space with uniform probability."""
        lower_bounds = np.argmax(mask == 0, axis=-1)
        masked_actions = self._rng.integers(low=lower_bounds, high=101) / 100
        if isinstance(masked_actions, float):
            masked_actions = [masked_actions]
        return masked_actions

    def _train_step(self, step: StateQuad):
        """
        Performs a single training step.

        Args:
            step: The step to train on.
        """
        st, at, rt, st_next = step.quad()
        # dims: (n_flavours, stock, reductions)
        self._G = rt + self._gamma * np.max(self._Q, axis=-1)[:, st_next]
        self._Q[:, st, at] += self._alpha * (self._G - self._Q[:, st, at])

    def train(self):
        self._env.reset()
        if self._warm_start is not None:
            logger.info(f"Warm starting for {self._warm_start} steps.")
            for _ in range(self._warm_start):
                self._env.reset()
                no_op = [0] * self._env.state_space_size[0]
                self._env.step(no_op)

        for epi in tqdm.tqdm(range(self._n_episodes)):
            if epi % 100 == 0:
                # logger.info(f"Episode {epi + 1}/{self._n_episodes}")
                pass

            env = deepcopy(self._env)
            st_0 = [product.stock for product in env.state.products.values()]
            is_terminal = False
            self._G = np.zeros(self._dims[0], dtype=np.float16)

            for step in range(self._horizon_steps):
                if is_terminal:
                    # logger.info("Reached terminal state, ending episode.")
                    break
                mask = env.mask_actions()
                a_i = self._select_action(current_stock=st_0, mask=mask)
                _, r_i, is_terminal, _ = env.step(a_i)
                r_i = [r for r in r_i.values()]
                st_i = [product.stock for product in env.state.products.values()]
                step = StateQuad(st_0, a_i, r_i, st_i)
                self._train_step(step)
                st_0 = st_i

            self._policy = np.round(np.argmax(self._Q, axis=-1) / 100, 2)
            self._rewards += [env.state.global_reward]
            self._discounted_rewards += [self._G.tolist()]
