import math
import numpy as np
from typing import Union, Tuple, Collection, Optional

from pricing.tools.utils import StatisticalProcess


class GaussianWhiteNoise(StatisticalProcess):
    """
    Independent observations drawn from Gaussian distributions :math:`N(0, \\sigma)`

    - Mean: the process mean is zero.
    - Variance: the process variance is constant across t.
    """

    __slots__ = ['sigma', '_u_t']

    def __init__(self, size: Tuple[int, ...], sigma: Union[float, Collection[float]] = 1.) -> None:
        """
        Method to initialize the Gaussian white noise process class with its relevant input.

        :param size: tuple containing the dimensions of the output statistical process. The first value represents
            the number of time steps simulated, the second number (if present) represents the number of independent
            paths simulated and the third number (if present) represents the number of different assets simulated.
        :param sigma: standard deviation of the Gaussian process.
        """
        super().__init__(size, None)
        self._assign_parameters_with_asset_dimension(sigma=sigma)
        object.__setattr__(self, '_u_t', None)

    def generate(self) -> np.ndarray:
        """
        Method to calculate the output for the statistical process.

        The output dimensions follow this schema:

        - First axis (necessary): number of time steps simulated.
        - Second axis (if present): number of independent paths simulated for one asset.
        - Third axis (if present): number of different assets simulated.

        :return: generated output object for the statistical process.
        """
        u_t = np.random.normal(loc=0, scale=self.sigma, size=self.size)
        object.__setattr__(self, '_u_t', u_t)
        return u_t
