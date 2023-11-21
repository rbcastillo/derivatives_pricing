import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm, multivariate_normal, lognorm
from math import sqrt
from typing import Union, Tuple, Collection, Optional, Any


class StatisticalProcess(ABC):
    """
    Base object to implement the main methods to be shared across all statistical processes implementations.
    """

    __slots__ = ['size', 'sub_periods', 'asset_attributes']

    @abstractmethod
    def __init__(self, size: Tuple[int, ...], sub_periods: Optional[int]) -> None:
        """
        Abstract method for the initialization of each specific object based on its implementation. The size
        attribute, needed across all statistical processes is loaded in this base method.

        The asset_attributes object is automatically created to store the name of the attributes that are
        associated to the asset dimension. This way, the appropriate review process is performed when these
        attributes are modified.

        :param size: tuple containing the dimensions of the output statistical process. The first value represents
            the number of time steps simulated, the second number (if present) represents the number of independent
            paths simulated and the third number (if present) represents the number of different assets simulated.
        :param sub_periods: number of sub-periods per reference time unit. The default is 252 trading days when one
            year is the reference time unit in which metrics such as returns and volatility are expressed.
        """
        object.__setattr__(self, 'size', self._manage_size(size))
        object.__setattr__(self, 'sub_periods', sub_periods)
        object.__setattr__(self, 'asset_attributes', [])

    @staticmethod
    def _manage_size(size: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Method to ensure that the size parameter is compliant with the logic of the process. The size parameter should
        meet the following criteria:

        - Length: the length of the parameter should be between 1 and 3 inclusive for it to have financial sense. The
          first dimension represents the number of time steps, the second the number of different paths and the third
          the number of asset simulations.
        - Zeros: the size tuple should not have zeros in the intermediate places, zeros at the end are removed reducing
          the length of the parameter.

        :param size: tuple containing the dimensions of the output statistical process.
        :return: adjusted tuple if needed, otherwise the original tuple is returned.
        """
        StatisticalProcess._manage_size_length(size)
        size = StatisticalProcess._manage_size_zeros(size)
        return size

    @staticmethod
    def _manage_size_length(size: Tuple[int, ...]) -> None:
        """
        Auxiliary method to ensure that the length of the size parameter is compliant with the logic of the process.
        The length of the parameter should be between 1 and 3 inclusive for it to have financial sense. The first
        dimension represents the number of time steps, the second the number of different paths and the third
        the number of asset simulations.

        :param size: tuple containing the dimensions of the output statistical process.
        :return: None.
        """
        if len(size) == 0:
            raise ValueError(f'Size <{size}> not valid, at least should have one dimension')
        elif len(size) > 3:
            raise ValueError(f'Size <{size}> not valid, maximum number of dimensions is 3')

    @staticmethod
    def _manage_size_zeros(size: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Method to ensure that the content of the size parameter is compliant with the logic of the process. The size
        tuple should not have zeros in the intermediate places, zeros at the end are removed reducing the length of
        the parameter.

        :param size: tuple containing the dimensions of the output statistical process.
        :return: adjusted tuple if needed, otherwise the original tuple is returned.
        """
        if 0 in size:
            if size[-1] == 0:
                warnings.warn('The last size dimension is zero, this dimension will be ignored')
                size = StatisticalProcess._manage_size(size[:-1])
            else:
                raise ValueError(f'Size <{size}> not valid, zero length inner dimensions are not allowed')
        return size

    def _assign_parameters_with_asset_dimension(self, **kwargs: Union[int, float, Collection]) -> None:
        """
        Method to assign the parameters that may vary by simulated asset after checking that they are compliant
        with the expected formats. If a parameter needs some adjustment, it is also managed inside this method.

        :param kwargs: keyword parameter(s) and value(s) to be verified and assigned.
        :return: None.
        """
        for name, value in kwargs.items():
            StatisticalProcess._check_parameter_with_asset_dimension(name, value, self.size)
            value = StatisticalProcess._adjust_parameter_with_asset_dimension(value, self.size)
            self.asset_attributes.append(name)
            object.__setattr__(self, name, value)

    @staticmethod
    def _check_parameter_with_asset_dimension(name: str, value: Union[int, float, Collection[Union[int, float]]],
                                              size: Tuple[int, ...]) -> None:
        """
        Auxiliary method to check, in case that an input applicable across assets is a collection, whether it meets
        the size requirements to be consistent with the target statistical process output size. If the parameter
        meets the requirements, the method does nothing. Otherwise, a ValueError exception is raised.

        :param name: name of the parameter to analyze.
        :param value: value of the parameter to analyze.
        :param size: target statistical process output size.
        :return: None.
        """
        if hasattr(value, '__len__') and not isinstance(value, str):
            if not len(size) == 3:
                raise ValueError(f'<{name}> is a collection but there is no asset dimension in size: <{size}>')
            elif len(value) != size[2]:
                raise ValueError(f'Length of <{name}> <{value}> does not match the asset dimension in size: <{size}>')

    @staticmethod
    def _adjust_parameter_with_asset_dimension(value: Union[int, float, Collection[Union[int, float]]],
                                               size: Tuple[int, ...]) -> np.ndarray:
        """
        Auxiliary method to check, in case that an input applicable across assets is a collection, whether it needs
        some adjustment. If the input is already a numpy ndarray, it is returned as is, if is a collection of
        other type, it is cast to ndarray and if it is a single numeric value, it is converted into a ndarray with
        a size matching the asset dimension.

        :param value: value of the parameter to analyze.
        :param size: target statistical process output size.
        :return: value object with the required adjustments.
        """
        if hasattr(value, '__len__'):
            value = value if isinstance(value, np.ndarray) else np.array(value)
        elif len(size) == 3:
            value = np.array([float(value)] * size[2])
        return value

    def _check_rho(self, rho: Optional[np.ndarray]) -> np.ndarray:
        """
        Method to check whether the provided correlation matrix rho meets all the mathematical properties of
        correlation matrices. The rho matrix is only used when there is asset dimension. If a rho matrix is
        provided without asset dimension, it will be ignored with a warning.

        If the rho matrix is required but not provided, an identity matrix is generated. Otherwise, the provided
        rho matrix is checked for inconsistencies.

        :param rho: optional input matrix containing the correlations between assets. It must be a square symmetric
            positive definite matrix with size equal to the number of assets in the asset dimension.
        :return: rho correlations matrix.
        """
        if len(self.size) == 3:
            rho = StatisticalProcess._manage_required_rho(rho=rho, n_assets=self.size[2])
        else:
            if rho is not None:
                warnings.warn('A rho input is provided but there is no asset dimension, rho will be ignored')
        return rho

    @staticmethod
    def _manage_required_rho(rho: Optional[np.ndarray], n_assets: int) -> np.ndarray:
        """
        Auxiliary method to manage the following cases when there is asset dimension and therefore, a rho matrix
        is required:

        - Matrix not provided. Independence of assets is assumed and every asset simulations are modelled
          independently of the rest of assets. As a consequence, an identity matrix is used as rho.
        - Matrix provided. The provided matrix is checked to ensure that it meets all the mathematical
          properties of a correlation matrix.

        :param rho: optional input matrix containing the correlations between assets. It must be a square symmetric
            positive definite matrix with size equal to the number of assets in the asset dimension.
        :param n_assets: number of assets defined by the size input.
        :return: rho correlations matrix.
        """
        if rho is None:
            rho = np.identity(n=n_assets)  # Equivalent to correlation matrix where all assets are independent
        else:
            StatisticalProcess._check_rho_properties(rho=rho, n_assets=n_assets)
        return rho

    @staticmethod
    def _check_rho_properties(rho: np.ndarray, n_assets: int) -> None:
        """
        Auxiliary method to assess that the input rho satisfies all the mathematical properties of a
        correlation matrix:

        - Shape. The rho matrix must be a square matrix and its dimension must match the number of assets.
        - Symmetry. The rho matrix must be symmetrical, since correlations are bidirectional.
        - Eigenvalues. The rho matrix must be definite positive, so (besides the symmetry condition already
          checked for) all its eigenvalues must be positive
        - Values. All values in the rho matrix must be within the closed interval [-1, 1], since this is the
          range of values for valid Pearson correlation coefficients.

        In case any of these properties is not met, an AssertionError is raised.

        :param rho: optional input matrix containing the correlations between assets. It must be a square symmetric
            positive definite matrix with size equal to the number of assets in the asset dimension.
        :param n_assets: number of assets defined by the size input.
        :return: None.
        """
        assert rho.shape == (n_assets, n_assets), f'rho shape <{rho.shape}> does not match n_assets: {n_assets}'
        assert np.array_equal(rho, rho.T), 'rho is not symmetric, not a valid correlation matrix'
        assert np.all(np.linalg.eigvals(rho) > 0), 'rho is not positive definite, not a valid correlation matrix'
        assert np.max(rho) <= 1 and np.min(rho) >= -1, 'rho values outside [-1, 1], not a valid correlation matrix'

    def __str__(self) -> str:
        """
        Method to create a string representation of the object. This method outputs the object name and the
        relevant parameters with their values.

        :return: string representation of the object.
        """
        attr_values = {attr: getattr(self, attr) for attr in self.__slots__ if not attr.startswith('_')}
        text_output = f'{self.__class__.__name__} object with parameters {attr_values}\n{self.__doc__}'
        return text_output

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Method to manage the process of setting and updating object parameters. Only attributes declared in the
        __init__ method that are not protected or private are allowed to be updated. This avoids adding
        new attributes or modifying attributes that should only be modified inside the object as a result of
        certain operations.

        :param key: parameter to be added or updated.
        :param value: value to be assigned to the parameter.
        :return: None
        """
        if key not in self.__slots__ or key.startswith('_'):
            valid_attrs = [attr for attr in self.__slots__ if not attr.startswith('_')]
            if key not in self.__slots__:
                raise ValueError(f'Attribute name <{key}> is not recognized, use values in {valid_attrs}')
            if key.startswith('_'):
                raise ValueError(f'Attribute <{key}> is protected or private, use values in {valid_attrs}')
        else:
            value = self._adjust_special_attributes_before_setting(key, value)
            object.__setattr__(self, key, value)

    def _adjust_special_attributes_before_setting(self, key: str, value: Any) -> Any:
        if key in self.asset_attributes:
            StatisticalProcess._check_parameter_with_asset_dimension(key, value, self.size)
            value = StatisticalProcess._adjust_parameter_with_asset_dimension(value, self.size)
        if key == 'rho':
            value = self._check_rho(value)
        return value

    def update_params(self, **kwargs: Any) -> None:
        """
        This method allows updating some or all of the relevant input for the European option class. The kwargs used
        in the method must match those in the __init__ method.

        :param kwargs: keyword parameter(s) and value(s) to be updated.
        :return: None.
        """
        for name, value in kwargs.items():
            setattr(self, name, value)

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Method to calculate the output for the statistical process, the implementation needs to be adjusted for each
        specific process in its class.

        The output dimensions follow this schema:

        - First axis (necessary): number of time steps simulated.
        - Second axis (if present): number of independent paths simulated for one asset.
        - Third axis (if present): number of different assets simulated.

        :return: generated output object for the statistical process.
        """
        pass


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

    __slots__ = ['rho', '_u_t', '_w_t']

    def __init__(self, size: Tuple[int, ...], rho: Optional[np.ndarray] = None, sub_periods: int = 252) -> None:
        """
        Method to initialize the Wiener process class with its relevant input.

        :param size: tuple containing the dimensions of the output statistical process. The first value represents
            the number of time steps simulated, the second number (if present) represents the number of independent
            paths simulated and the third number (if present) represents the number of different assets simulated.
        :param rho: optional input matrix containing the correlations between assets. It must be a square symmetric
            positive definite matrix with size equal to the number of assets in the asset dimension.
        :param sub_periods: number of sub-periods per reference time unit. The default is 252 trading days when one
            year is the reference time unit in which metrics such as returns and volatility are expressed.
        """
        super().__init__(size, sub_periods)
        object.__setattr__(self, 'rho', self._check_rho(rho))
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
        u_t = GaussianWhiteNoise(size=self.size, sigma=1 / sqrt(self.sub_periods)).generate()
        object.__setattr__(self, '_u_t', u_t)
        w_0 = np.zeros(shape=(1,) + self.size[1:])  # Set W_0 = 0 according to Wiener process properties
        w_t = np.concatenate((w_0, u_t.cumsum(axis=0)), axis=0)
        if len(self.size) == 3:  # If there is an asset dimension, manage the correlation between processes
            # Cholesky decomposition makes rho a lower triangular matrix so that correlation is built incrementally
            w_t = np.tensordot(w_t, np.linalg.cholesky(self.rho), axes=(-1, -1))
        object.__setattr__(self, '_w_t', w_t)
        return w_t

    def generate_distribution(self, num_points: int = 500_000) -> np.ndarray:
        """
        Method to calculate the output distribution at the last time step for the statistical process.

        The output dimensions follow this schema:

        - If there is no asset dimension, resulting uni-variate distribution for a single asset with
          num_points values. In the uni-variate version the output covers the ordered x values for the
          inverse CDF of probabilities in (1e-10, 1 - 1e-10).
        - If there is an asset dimension, resulting multi-variate distribution for the number of assets
          defined in the size attribute with (num_points x num_assets) values. In the multi-variate version
          the output is obtained via random sampling of the joint distribution.

        :param num_points: number of points used to generate the statistical distribution for the process.
        :return: generated output object for the statistical process.
        """
        mu, sigma = 0, np.sqrt(self.size[0] / self.sub_periods)
        if len(self.size) < 3:
            x_space = np.linspace(1e-10, 1 - 1e-10, num_points)
            w_t = norm(loc=mu, scale=sigma).ppf(x_space)
        else:
            w_t = multivariate_normal(mean=[mu] * self.size[2], cov=self.rho * sigma**2).rvs(num_points)
        return w_t


class GeometricBrownianMotion(StatisticalProcess):
    """
    Geometric Brownian motion :math:`\\small \\displaystyle S_t = S_0 e^{(\\mu-q-0.5\\sigma^2)t+\\sigma W_t}`,
    where :math:`\\small \\displaystyle W_t` is a Wiener process or ordinary Brownian motion.

    - Mean: the process mean at time step t is :math:`\\small \\displaystyle S_0 e^{(\\mu - q) t}`.
    - Variance: the process variance at time step t is :math:`\\small \\displaystyle S_0^2 e^{2(\\mu - q) t}
      (e^{\\sigma^{2} t} - 1)`.
    """

    __slots__ = ['s0', 'mu', 'sigma', 'q', 'rho', '_w_t', '_s_t']

    def __init__(self, size: Tuple[int, ...], s0: Union[float, Collection[float]], mu: Union[float, Collection[float]],
                 sigma: Union[float, Collection[float]], q: Union[float, Collection[float]] = 0,
                 rho: Optional[np.ndarray] = None, sub_periods: int = 252) -> None:
        """
        Method to initialize the Wiener process class with its relevant input.

        :param size: tuple containing the dimensions of the output statistical process. The first value represents
            the number of time steps simulated, the second number (if present) represents the number of independent
            paths simulated and the third number (if present) represents the number of different assets simulated.
        :param sigma: standard deviation of the Gaussian process.
        :param rho: optional input matrix containing the correlations between assets. It must be a square symmetric
            positive definite matrix with size equal to the number of assets in the asset dimension.
        :param sub_periods: number of sub-periods per reference time unit. The default is 252 trading days when one
            year is the reference time unit in which metrics such as returns and volatility are expressed.
        """
        super().__init__(size, sub_periods)
        self._assign_parameters_with_asset_dimension(s0=s0, mu=mu, sigma=sigma, q=q)
        object.__setattr__(self, 'rho', rho)
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
        w_t = Wiener(size=self.size, rho=self.rho, sub_periods=self.sub_periods).generate()
        object.__setattr__(self, '_w_t', w_t)
        t = np.arange(0, self.size[0] + 1) / self.sub_periods
        for _ in range(1, len(self.size)):  # Adjust t dimensions
            t = t[:, np.newaxis]
        drift = (self.mu - self.q - .5 * self.sigma ** 2) * t  # Model drift: (μ-q-0.5·σ^2)t
        s_t = self.s0 * np.exp(drift + self.sigma * w_t)  # S_t = S_0·exp( (μ-q-0.5·σ^2)t + σ·W_t)
        object.__setattr__(self, '_s_t', s_t)
        return s_t

    def generate_distribution(self, num_points: int = 500_000) -> np.ndarray:
        """
        Method to calculate the output distribution at the last time step for the statistical process.

        The output dimensions follow this schema:

        - If there is no asset dimension, resulting uni-variate distribution for a single asset with
          num_points values. In the uni-variate version the output covers the ordered x values for the
          inverse CDF of probabilities in (1e-10, 1 - 1e-10).
        - If there is an asset dimension, resulting multi-variate distribution for the number of assets
          defined in the size attribute with (num_points x num_assets) values. In the multi-variate version
          the output is obtained via random sampling of the joint distribution.

        :param num_points: number of points used to generate the statistical distribution for the process.
        :return: generated output object for the statistical process.
        """
        time_units = self.size[0] / self.sub_periods
        process_sigma = self.sigma * np.sqrt(time_units)
        process_scale = self.s0 * np.exp((self.mu - self.q - .5 * self.sigma**2) * time_units)
        if len(self.size) < 3:
            x_space = np.linspace(1e-10, 1 - 1e-10, num_points)
            s_t = lognorm(s=process_sigma, scale=process_scale).ppf(x_space)
        else:
            w_t = Wiener(self.size, self.rho, self.sub_periods).generate_distribution(num_points)
            drift = (self.mu - self.q - .5 * self.sigma ** 2) * time_units
            s_t = self.s0 * np.exp(drift + self.sigma * w_t)
        return s_t
