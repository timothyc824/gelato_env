import torch
from typing import Tuple

from data_generators.generator import Generator


class NaiveGaussian(Generator):

    def __init__(self, mean: float = 100, std: float = 10):

        self._distribution = torch.distributions.Normal(loc=mean, scale=std)

    def sample(self, n: Tuple[int, ...] = (1,)) -> torch.Tensor:
        return torch.clip(self._distribution.sample(n).int(), min=0)


class WeakSeasonalGaussian(Generator):

    def __init__(self, mean: float = 100, std: float = 10):

        self._distribution = torch.distributions.Normal(loc=mean, scale=std)
        self._scaling = torch.distributions.Beta(concentration0=7, concentration1=10)

    def sample(self, n: Tuple[int, ...] = (1,)) -> torch.Tensor:
        background_dist = self._distribution.sample(n)
        time_values = torch.linspace(0, 1, n[-1])
        scaling = torch.exp(self._scaling.log_prob(time_values))
        return torch.clip((scaling * background_dist).int(), min=0)


class StrongSeasonalGaussian(Generator):

    def __init__(self, mean: float = 100, std: float = 10):

        self._distribution = torch.distributions.Normal(loc=mean, scale=std)
        self._scaling = torch.distributions.Beta(concentration0=7, concentration1=10)
        self._weekly_seasonality = torch.distributions.Beta(concentration0=7,
                                                            concentration1=10)

    def sample(self, n: Tuple[int, ...] = (1,)) -> torch.Tensor:
        background_dist = self._distribution.sample(n)
        time_values = torch.linspace(0, 1, 365).tile(max(1, n[-1] // 365))
        weekly_values = torch.arange(0, 1, 1/7)
        yearly_values = torch.cat([weekly_values.repeat(n[-1] // 7),
                                   weekly_values[:n[-1] % 7]])
        scaling = torch.exp(self._scaling.log_prob(time_values))
        weekly_scaling = .005 * torch.exp(self._weekly_seasonality.log_prob(yearly_values))
        return torch.clip(((scaling + weekly_scaling) * background_dist).int(), min=0)
