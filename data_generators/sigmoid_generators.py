import torch

from data_generators.generator import Generator


class SigmoidGaussian(Generator):
    """
    A class that generates a sigmoid function with a gaussian noise component. Used to generate probabilities of sales
    of a product at a given markdown.
    """

    def __init__(self, mean: float = .1, std: float = .05):
        self._background = lambda x: 1 + torch.sigmoid(5 * x - 2.5)
        self._noise = torch.distributions.Normal(loc=mean, scale=std)

    def prob_sales_at_reduction(self, reductions: torch.Tensor) -> torch.Tensor:
        return torch.clip((self._noise.sample(sample_shape=reductions.shape) + self._background(reductions)),
                          min=0, max=2)
