from math import sqrt
from typing import Union, Tuple, Optional, Callable, List

import numpy as np
from scipy.stats import norm, chi2

from pricing.simulation.processes import Wiener, GeometricBrownianMotion
from tests.env_variables import ALPHA, HIGH_TOL, REL_TOL


class HelperTestUtils:

    @staticmethod
    def adjust_single_parameter(process: np.ndarray, parameter: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if len(process.shape) == 3:
            if not hasattr(parameter, '__len__'):
                parameter = np.array([parameter] * process.shape[2])
            elif not isinstance(parameter, np.ndarray):
                parameter = np.array(parameter)
        return parameter

    @staticmethod
    def adjust_all_parameters(process: np.ndarray, *args) -> List[Union[float, np.ndarray]]:
        adjusted_args = [HelperTestUtils.adjust_single_parameter(process, arg) for arg in args]
        return adjusted_args


class HelperDistributions:

    @staticmethod
    def calc_process_mean_distribution(process: np.ndarray) -> np.ndarray:
        distribution = process
        return distribution

    @staticmethod
    def calc_process_std_distribution(process: np.ndarray) -> np.ndarray:
        distribution = process
        return distribution

    @staticmethod
    def calc_process_rho_distribution(process: np.ndarray) -> np.ndarray:
        returns = np.diff(process, axis=0)
        distribution = np.array([np.corrcoef(returns[:, i, :], rowvar=False) for i in range(returns.shape[1])])
        return distribution

    @staticmethod
    def calc_performance_distribution(process: np.ndarray) -> np.ndarray:
        distribution = process[-1, :] / process[0, :] - 1
        return distribution

    @staticmethod
    def calc_returns_volatility_distribution(process: np.ndarray) -> np.ndarray:
        returns = np.log(process[1:] / process[:-1])
        distribution = returns.std(axis=0, ddof=1)
        return distribution


class HelperConfidenceIntervals:

    @staticmethod
    def calc_mean_interval(mu: Union[float, np.ndarray], sigma: Union[float, np.ndarray],
                           n: int) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """
        Confidence interval calculation for the mean estimator when samples come from a Gaussian distribution with
        known variance using the following formula:

        .. math::

            \\small \\displaystyle \\overline{x} - Z_{1-\\frac{\\alpha}{2}} \\frac{\\sigma}{\\sqrt{n}} \\le \\mu
            \\le \\overline{x} + Z_{1-\\frac{\\alpha}{2}} \\frac{\\sigma}{\\sqrt{n}}

        """
        z_score: float = norm(loc=0, scale=1).ppf(1 - ALPHA / 2)
        mean_error = z_score * sigma / sqrt(n)
        lower, upper = mu - mean_error, mu + mean_error
        return lower, upper

    @staticmethod
    def calc_std_interval(sigma: Union[float, np.ndarray],
                          n: int) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """
        Confidence interval calculation for the standard deviation estimator when samples come from a Gaussian
        distribution with known mean using the following formula:

        .. math::

            \\small \\displaystyle \\frac{(n - 1)s^2}{\\chi(n - 1)_{1-\\frac{\\alpha}{2}}} \\le \\sigma^2
            \\le \\frac{(n - 1)s^2}{\\chi(n - 1)_{\\frac{\\alpha}{2}}}

        """
        lower = np.sqrt((n - 1) * sigma ** 2 / chi2(n - 1).ppf(1 - ALPHA / 2))
        upper = np.sqrt((n - 1) * sigma ** 2 / chi2(n - 1).ppf(ALPHA / 2))
        return lower, upper

    @staticmethod
    def calc_rho_interval(rho: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Confidence interval calculation for the Pearson's correlation coefficient estimator based on the Fisher
        transformation using the following formula:

        .. math::

            \\small \\displaystyle tanh(\\frac{arctanh(rho) - Z_{1-\\frac{\\alpha}{2}}}{\\sqrt{n-3}}) \\le \\rho
            \\le tanh(\\frac{arctanh(rho) + Z_{1-\\frac{\\alpha}{2}}}{\\sqrt{n-3}})

        """
        fisher_rho = np.arctanh(rho.copy() - np.identity(n=len(rho)))
        fisher_error = norm(loc=0, scale=1).ppf(1 - ALPHA / 2) / sqrt(n - 3)
        lower, upper = np.tanh(fisher_rho - fisher_error), np.tanh(fisher_rho + fisher_error)
        np.fill_diagonal(lower, -1e-10), np.fill_diagonal(upper, 1e-10)
        return lower, upper


class HelperStatisticalTests:

    @staticmethod
    def _get_interval_test_ground_truth(tail_pct: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        contrast = ALPHA
        if (len(tail_pct.shape) > 1) and (tail_pct.shape[0] == tail_pct.shape[1]):
            contrast = np.full(tail_pct.shape, ALPHA) + np.identity(len(tail_pct)) * (1 - ALPHA)
        return contrast

    @staticmethod
    def test_interval(distribution: np.ndarray, lower: Union[float, np.ndarray],
                      upper: Union[float, np.ndarray]) -> bool:
        tail_pct = ((distribution <= lower) | (distribution >= upper)).mean(axis=0)
        contrast = HelperStatisticalTests._get_interval_test_ground_truth(tail_pct)
        interval_result = np.allclose(tail_pct, contrast, atol=HIGH_TOL)
        return interval_result

    @staticmethod
    def test_absolute(distribution: np.ndarray, ground_truth: Union[float, np.ndarray]) -> bool:
        absolute_result = np.allclose(distribution.mean(axis=0), ground_truth, atol=HIGH_TOL, rtol=REL_TOL)
        return absolute_result

    @staticmethod
    def test_metric(process: np.ndarray, distribution_method: Callable, interval_method: Callable,
                    interval_kwargs: dict, expected_absolute: Union[float, np.ndarray]) -> bool:
        distribution = distribution_method(process)
        lower, upper = interval_method(**interval_kwargs)
        interval_result = HelperStatisticalTests.test_interval(distribution, lower, upper)
        absolute_result = HelperStatisticalTests.test_absolute(distribution, expected_absolute)
        return all([interval_result, absolute_result])

    @staticmethod
    def test_metric_absolute_only(process: np.ndarray, distribution_method: Callable,
                                  expected_absolute: Union[float, np.ndarray]) -> bool:
        distribution = distribution_method(process)
        absolute_result = HelperStatisticalTests.test_absolute(distribution, expected_absolute)
        return absolute_result


class HelperRunTests:

    @staticmethod
    def run_shape_test(process: np.ndarray, size: tuple) -> bool:
        if process[0].sum() == 0:
            size = (size[0] + 1,) + size[1:]
        shape_test = process.shape == size
        return shape_test

    @staticmethod
    def run_process_mean_test(process: np.ndarray, mu: Union[float, np.ndarray], sigma: Union[float, np.ndarray],
                              n: int) -> bool:
        distribution_method = HelperDistributions.calc_process_mean_distribution
        interval_method = HelperConfidenceIntervals.calc_mean_interval
        interval_kwargs = {'mu': mu, 'sigma': sigma, 'n': n}
        # Pre-calculate mean unless we extract only means from the process
        process = process if n not in process.shape else process.mean(axis=0)
        test_result = HelperStatisticalTests.test_metric(process, distribution_method, interval_method,
                                                         interval_kwargs, expected_absolute=mu)
        return test_result

    @staticmethod
    def run_process_std_test(process: np.ndarray, sigma: Union[float, np.ndarray], n: int) -> bool:
        distribution_method = HelperDistributions.calc_process_std_distribution
        interval_method = HelperConfidenceIntervals.calc_std_interval
        interval_kwargs = {'sigma': sigma, 'n': n}
        # Pre-calculate standard deviation unless we extract only standard deviations from the process
        process = process if n not in process.shape else process.std(axis=0)
        test_result = HelperStatisticalTests.test_metric(process, distribution_method, interval_method,
                                                         interval_kwargs, expected_absolute=sigma)
        return test_result

    @staticmethod
    def run_process_std_absolute_test(process: np.ndarray, sigma: Union[float, np.ndarray], n: int) -> bool:
        distribution_method = HelperDistributions.calc_process_std_distribution
        # Pre-calculate standard deviation unless we extract only standard deviations from the process
        process = process if n not in process.shape else process.std(axis=0)
        test_result = HelperStatisticalTests.test_metric_absolute_only(process, distribution_method,
                                                                       expected_absolute=sigma)
        return test_result

    @staticmethod
    def run_process_rho_test(process: np.ndarray, rho: np.ndarray, n: int) -> bool:
        test_result = True
        if len(process) == 3:
            rho = np.identity(3) if rho is None else rho
            distribution_method = HelperDistributions.calc_process_rho_distribution
            interval_method = HelperConfidenceIntervals.calc_rho_interval
            interval_kwargs = {'rho': rho, 'n': n}
            test_result = HelperStatisticalTests.test_metric(process, distribution_method, interval_method,
                                                             interval_kwargs, expected_absolute=rho)
        return test_result

    @staticmethod
    def run_cumulative_performance_test(process: np.ndarray, mu: Union[float, np.ndarray],
                                        q: Union[float, np.ndarray], time_units: int) -> bool:
        distribution_method = HelperDistributions.calc_performance_distribution
        expected_performance = np.exp((mu - q) * time_units) - 1
        test_result = HelperStatisticalTests.test_metric_absolute_only(process, distribution_method,
                                                                       expected_absolute=expected_performance)
        return test_result

    @staticmethod
    def run_returns_volatility_test(process: np.ndarray, sigma: Union[float, np.ndarray], sub_periods: int) -> bool:
        distribution_method = HelperDistributions.calc_returns_volatility_distribution
        test_result = HelperStatisticalTests.test_metric_absolute_only(process, distribution_method,
                                                                       expected_absolute=sigma / sqrt(sub_periods))
        return test_result


class HelperGaussianWhiteNoise(HelperStatisticalTests):

    @staticmethod
    def calc_tests(process: np.ndarray, size: tuple, mu: Union[float, List[float]],
                   sigma: Union[float, List[float]]) -> List[bool]:
        mu, sigma = HelperTestUtils.adjust_all_parameters(process, mu, sigma)
        test_input = {'mean': {'mu': 0., 'sigma': sigma, 'n': process.shape[0]},
                      'std': {'sigma': sigma, 'n': process.shape[0]}}
        test_results = [HelperRunTests.run_process_mean_test(process, **test_input['mean']),
                        HelperRunTests.run_process_std_test(process, **test_input['std']),
                        HelperRunTests.run_shape_test(process, size)]
        return test_results


class HelperWiener(HelperStatisticalTests):

    @staticmethod
    def calc_tests(w: Wiener, process: np.ndarray, size: tuple, time_units: int, num_paths: int,
                   rho: Optional[np.ndarray] = None) -> List[bool]:
        stats = zip(*[(lambda x: (x.mean(axis=0), x.std(axis=0, ddof=1)))(w.generate()[-1]) for _ in range(10_000)])
        process_mean, process_std = map(np.array, stats)
        test_input = {'mean': {'mu': 0., 'sigma': sqrt(time_units), 'n': num_paths},
                      'std': {'sigma': sqrt(time_units), 'n': num_paths},
                      'rho': {'rho': rho, 'n': process.shape[0]}}
        test_results = [HelperRunTests.run_process_mean_test(process_mean, **test_input['mean']),
                        HelperRunTests.run_process_std_test(process_std, **test_input['std']),
                        HelperRunTests.run_process_rho_test(process, **test_input['rho']),
                        HelperRunTests.run_shape_test(process, size)]
        return test_results


class HelperGBM(HelperStatisticalTests):

    @staticmethod
    def calc_tests(gbm: GeometricBrownianMotion, process: np.ndarray, size: tuple, time_units: int, num_paths: int,
                   mu: Union[float, List[float]], sigma: Union[float, List[float]], q: Union[float, List[float]],
                   s0: Union[float, List[float]], rho: Optional[np.ndarray] = None) -> List[bool]:

        s0, mu, sigma, q = HelperTestUtils.adjust_all_parameters(process, s0, mu, sigma, q)

        stats = zip(*[(lambda x: (x.mean(axis=0), x.std(axis=0, ddof=1)))(gbm.generate()[-1]) for _ in range(10_000)])
        process_mean, process_std = map(np.array, stats)
        gbm_mean = s0 * np.exp((mu - q) * time_units)
        gbm_std = np.sqrt(s0 ** 2 * np.exp(2 * (mu - q) * time_units) * (np.exp(time_units * sigma ** 2) - 1))
        test_input = {'mean': {'mu': gbm_mean, 'sigma': gbm_std, 'n': num_paths},
                      'std': {'sigma': gbm_std, 'n': num_paths},
                      'rho': {'rho': rho, 'n': process.shape[0]}}
        test_results = [HelperRunTests.run_process_mean_test(process_mean, **test_input['mean']),
                        HelperRunTests.run_process_std_absolute_test(process_std, **test_input['std']),
                        HelperRunTests.run_process_rho_test(process, **test_input['rho']),
                        HelperRunTests.run_shape_test(process - s0, size),
                        HelperRunTests.run_cumulative_performance_test(process, mu, q, time_units),
                        HelperRunTests.run_returns_volatility_test(process, sigma, gbm.sub_periods)]

        return test_results
