import unittest
from math import sqrt, exp

import numpy as np

from pricing.simulation.processes import GaussianWhiteNoise, Wiener, GeometricBrownianMotion
from tests.env_variables import LOW_TOL, HIGH_TOL


class TestGaussianWhiteNoise(unittest.TestCase):

    def test_gaussian_white_noise_1d(self) -> None:
        size = (500_000, )
        sigma = .5
        process = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        shape_test = process.shape == size
        mean_test = np.isclose(process.mean(), 0., atol=LOW_TOL)
        std_test = np.isclose(process.std(), sigma, atol=LOW_TOL)
        self.assertTrue(shape_test and mean_test and std_test)

    def test_gaussian_white_noise_2d(self) -> None:
        size = (100_000, 1_000)
        sigma = .5
        process = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        shape_test = process.shape == size
        mean_test = np.isclose(process.mean(), 0., atol=LOW_TOL)
        std_test = np.isclose(process.std(axis=0).mean(), sigma, atol=LOW_TOL)
        self.assertTrue(shape_test and mean_test and std_test)

    def test_gaussian_white_noise_3d_single_sigma(self) -> None:
        size = (100_000, 1_000, 3)
        sigma = .5
        process = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        shape_test = process.shape == size
        mean_test = np.isclose(process.mean(), 0., atol=LOW_TOL)
        std_test = np.allclose(process.std(axis=0).mean(axis=0), np.array([sigma] * size[2]), atol=LOW_TOL)
        self.assertTrue(shape_test and mean_test and std_test)

    def test_gaussian_white_noise_3d_vector_sigma(self) -> None:
        size = (100_000, 1_000, 3)
        sigma = [.25, .5, 1.]
        process = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        shape_test = process.shape == size
        mean_test = np.isclose(process.mean(), 0., atol=LOW_TOL)
        std_test = np.allclose(process.std(axis=0).mean(axis=0), np.array(sigma), atol=LOW_TOL)
        self.assertTrue(shape_test and mean_test and std_test)


class TestWiener(unittest.TestCase):

    def test_wiener_2d(self) -> None:
        periods = 12
        num_periods = 10
        size = (periods * num_periods, 500_000)
        process = Wiener(size=size, periods=periods).generate()
        shape_test = process.shape == (size[0] + 1, ) + size[1:]
        mean_test = np.isclose(process[-1, :].mean(), 0., atol=LOW_TOL)
        std_test = np.isclose(process[-1, :].std(axis=0).mean(), sqrt(num_periods), atol=LOW_TOL)
        self.assertTrue(shape_test and mean_test and std_test)

    def test_wiener_3d_no_correlation(self) -> None:
        periods = 12
        num_periods = 10
        num_assets = 3
        size = (periods * num_periods, 500_000, num_assets)
        process = Wiener(size=size, periods=periods).generate()
        shape_test = process.shape == (size[0] + 1, ) + size[1:]
        mean_test = np.isclose(process[-1, :].mean(), 0., atol=LOW_TOL)
        std_test = np.isclose(process[-1, :].std(axis=0).mean(), sqrt(num_periods), atol=LOW_TOL)
        correlations = np.corrcoef(process.reshape(-1, process.shape[-1]).T)
        corr_test = np.allclose(correlations, np.identity(n=num_assets), atol=LOW_TOL)
        self.assertTrue(shape_test and mean_test and std_test and corr_test)

    def test_wiener_3d_with_correlation(self) -> None:
        periods = 12
        num_periods = 10
        num_assets = 3
        size = (periods * num_periods, 500_000, num_assets)
        rho = np.array([[1., .25, .5], [.25, 1., .75], [.5, .75, 1.]])
        process = Wiener(size=size, rho=rho, periods=periods).generate()
        shape_test = process.shape == (size[0] + 1, ) + size[1:]
        mean_test = np.isclose(process[-1, :].mean(), 0., atol=LOW_TOL)
        std_test = np.allclose(process[-1, :].std(axis=0), sqrt(num_periods), atol=LOW_TOL)
        correlations = np.corrcoef(process.reshape(-1, process.shape[-1]).T)
        corr_test = np.allclose(correlations, rho, atol=LOW_TOL)
        self.assertTrue(shape_test and mean_test and std_test and corr_test)


class TestGeometricBrownianMotion(unittest.TestCase):

    def test_gbm_2d(self) -> None:
        periods = 12
        num_periods = 5
        size = (periods * num_periods, 500_000)
        s0, mu, sigma, q = 100, .15, .20, .05
        process = GeometricBrownianMotion(size=size, s0=s0, mu=mu, sigma=sigma, q=q, periods=periods).generate()
        shape_test = process.shape == (size[0] + 1, ) + size[1:]
        expected_mean = s0 * np.exp((mu - q) * num_periods)
        mean_test = np.isclose(process[-1, :].mean(), expected_mean, atol=HIGH_TOL)
        expected_std = sqrt(s0 ** 2 * exp(2 * (mu - q) * num_periods) * (exp(num_periods * sigma ** 2) - 1))
        std_test = np.isclose(process[-1, :].std(axis=0).mean(), expected_std, atol=HIGH_TOL)
        returns = process[1:, :] / process[:-1, :] - 1
        performance = process[-1, :] / process[0, :] - 1
        mu_test = np.isclose(performance.mean(), exp((mu - q) * num_periods) - 1, atol=LOW_TOL)
        vol_test = np.isclose(returns.std(axis=0).mean() * sqrt(periods), sigma, atol=LOW_TOL)
        self.assertTrue(shape_test and mean_test and std_test and mu_test and vol_test)

    def test_wiener_3d_no_correlation(self) -> None:
        periods = 12
        num_periods = 5
        num_assets = 3
        size = (periods * num_periods, 500_000, num_assets)
        s0, mu, sigma, q = 100, .15, .20, .05
        process = GeometricBrownianMotion(size=size, s0=s0, mu=mu, sigma=sigma, q=q, periods=periods).generate()
        shape_test = process.shape == (size[0] + 1, ) + size[1:]
        expected_mean = s0 * np.exp((mu - q) * num_periods)
        mean_test = np.isclose(process[-1, :].mean(), expected_mean, atol=HIGH_TOL)
        expected_std = sqrt(s0 ** 2 * exp(2 * (mu - q) * num_periods) * (exp(num_periods * sigma ** 2) - 1))
        std_test = np.isclose(process[-1, :].std(axis=0).mean(), expected_std, atol=HIGH_TOL)
        returns = process[1:, :, :] / process[:-1, :, :] - 1
        performance = process[-1, :] / process[0, :] - 1
        mu_test = np.isclose(performance.mean(), exp((mu - q) * num_periods) - 1, atol=LOW_TOL)
        vol_test = np.isclose(returns.std(axis=0).mean() * sqrt(periods), sigma, atol=LOW_TOL)
        correlations = np.corrcoef(returns.reshape(-1, returns.shape[-1]).T)
        corr_test = np.allclose(correlations, np.identity(n=num_assets), atol=LOW_TOL)
        self.assertTrue(shape_test and mean_test and std_test and mu_test and vol_test and corr_test)

    def test_wiener_3d_with_correlation(self) -> None:
        periods = 12
        num_periods = 5
        num_assets = 3
        size = (periods * num_periods, 500_000, num_assets)
        s0, mu, sigma, q = 100, .15, .20, .05
        rho = np.array([[1., .25, .5], [.25, 1., .75], [.5, .75, 1.]])
        np.random.seed(42)
        process = GeometricBrownianMotion(size=size, s0=s0, mu=mu, sigma=sigma, q=q, rho=rho,
                                          periods=periods).generate()
        shape_test = process.shape == (size[0] + 1, ) + size[1:]
        expected_mean = s0 * np.exp((mu - q) * num_periods)
        mean_test = np.isclose(process[-1, :].mean(), expected_mean, atol=1)
        expected_std = sqrt(s0 ** 2 * exp(2 * (mu - q) * num_periods) * (exp(num_periods * sigma ** 2) - 1))
        std_test = np.isclose(process[-1, :].std(axis=0).mean(), expected_std, atol=1)
        returns = process[1:, :, :] / process[:-1, :, :] - 1
        performance = process[-1, :] / process[0, :] - 1
        mu_test = np.isclose(performance.mean(), exp((mu - q) * num_periods) - 1, atol=LOW_TOL)
        vol_test = np.isclose(returns.std(axis=0).mean() * sqrt(periods), sigma, atol=LOW_TOL)
        correlations = np.corrcoef(returns.reshape(-1, returns.shape[-1]).T)
        corr_test = np.allclose(correlations, rho, atol=LOW_TOL)
        self.assertTrue(shape_test and mean_test and std_test and mu_test and vol_test and corr_test)

    def test_wiener_3d_with_correlation_vector_variables(self) -> None:
        periods = 12
        num_periods = 5
        num_assets = 3
        size = (periods * num_periods, 500_000, num_assets)
        s0, mu, sigma, q = [100, 150, 50], [.15, .10, .05], [.20, .10, .15], [.05, .02, .01]
        rho = np.array([[1., .25, .5], [.25, 1., .75], [.5, .75, 1.]])
        np.random.seed(42)
        process = GeometricBrownianMotion(size=size, s0=s0, mu=mu, sigma=sigma, q=q, rho=rho,
                                          periods=periods).generate()
        s0, mu, sigma, q = np.array(s0), np.array(mu), np.array(sigma), np.array(q)
        shape_test = process.shape == (size[0] + 1, ) + size[1:]
        expected_mean = s0 * np.exp((mu - q) * num_periods)
        mean_test = np.allclose(process[-1, :].mean(axis=0), expected_mean, atol=1)
        expected_std = np.sqrt(s0 ** 2 * np.exp(2 * (mu - q) * num_periods) * (np.exp(num_periods * sigma ** 2) - 1))
        std_test = np.allclose(process[-1, :].std(axis=0), expected_std, atol=1)
        returns = process[1:, :, :] / process[:-1, :, :] - 1
        performance = process[-1, :] / process[0, :] - 1
        mu_test = np.allclose(performance.mean(axis=0), np.exp((mu - q) * num_periods) - 1, atol=LOW_TOL)
        vol_test = np.allclose(returns.std(axis=0).mean(axis=0) * sqrt(periods), sigma, atol=LOW_TOL)
        correlations = np.corrcoef(returns.reshape(-1, returns.shape[-1]).T)
        corr_test = np.allclose(correlations, rho, atol=LOW_TOL)
        self.assertTrue(shape_test and mean_test and std_test and mu_test and vol_test and corr_test)


if __name__ == '__main__':
    unittest.main()
