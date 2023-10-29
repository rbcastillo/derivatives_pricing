import numpy as np
from typing import Union, Tuple, Collection, Optional
from math import sqrt

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


class Wiener(StatisticalProcess):
    """
    Wiener process or Brownian motion :math:`W_t`.

    - Mean: the process mean is zero for every t.
    - Variance: the process variance at time step t is t.
    """

    __slots__ = ['sigma', 'rho', '_u_t', '_w_t']

    def __init__(self, size: Tuple[int, ...], rho: Optional[np.ndarray] = None, periods: int = 252) -> None:
        """
        Method to initialize the Wiener process class with its relevant input.

        :param size: tuple containing the dimensions of the output statistical process. The first value represents
            the number of time steps simulated, the second number (if present) represents the number of independent
            paths simulated and the third number (if present) represents the number of different assets simulated.
        :param rho: optional input matrix containing the correlations between assets. It must be a square symmetric
            positive definite matrix with size equal to the number of assets in the asset dimension.
        :param periods: number of periods per reference time unit. The default is 252 trading days when one year is
            the reference time unit in which metrics such as returns and volatility are expressed.
        """
        super().__init__(size, periods)
        object.__setattr__(self, 'rho', self._check_and_adjust_rho(rho))
        object.__setattr__(self, '_u_t', None)
        object.__setattr__(self, '_w_t', None)

    def generate(self) -> np.ndarray:
        """
        Method to calculate the output for the statistical process.

        The output dimensions follow this schema:

        - First axis (necessary): number of time steps simulated.
        - Second axis (if present): number of independent paths simulated for one asset.
        - Third axis (if present): number of different assets simulated.

        :return: generated output object for the statistical process.
        """
        u_t = GaussianWhiteNoise(size=self.size, sigma=1 / sqrt(self.periods)).generate()
        object.__setattr__(self, '_u_t', u_t)
        w_0 = np.zeros(shape=(1,) + self.size[1:])  # Set W_0 = 0 according to Wiener process properties
        w_t = np.concatenate((w_0, u_t.cumsum(axis=0)), axis=0)
        if len(self.size) == 3:  # If there is an asset dimension, manage the correlation between processes
            w_t = np.tensordot(w_t, self.rho, axes=(-1, -1))
        object.__setattr__(self, '_w_t', w_t)
        return w_t
