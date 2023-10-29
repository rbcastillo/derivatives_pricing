import unittest
from math import sqrt

import numpy as np

from pricing.simulation.processes import GaussianWhiteNoise, Wiener
from tests.env_variables import TOLERANCE


class TestGaussianWhiteNoise(unittest.TestCase):

    def test_gaussian_white_noise_1d(self) -> None:
        size = (500_000, )
        sigma = .5
        process = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        shape_test = process.shape == size
        mean_test = np.isclose(process.mean(), 0., atol=TOLERANCE)
        std_test = np.isclose(process.std(), sigma, atol=TOLERANCE)
        self.assertTrue(shape_test and mean_test and std_test)

    def test_gaussian_white_noise_2d(self) -> None:
        size = (100_000, 1_000)
        sigma = .5
        process = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        shape_test = process.shape == size
        mean_test = np.isclose(process.mean(), 0., atol=TOLERANCE)
        std_test = np.isclose(process.std(axis=0).mean(), sigma, atol=TOLERANCE)
        self.assertTrue(shape_test and mean_test and std_test)

    def test_gaussian_white_noise_3d_single_sigma(self) -> None:
        size = (100_000, 1_000, 3)
        sigma = .5
        process = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        shape_test = process.shape == size
        mean_test = np.isclose(process.mean(), 0., atol=TOLERANCE)
        std_test = np.allclose(process.std(axis=0).mean(axis=0), np.array([sigma] * size[2]), atol=TOLERANCE)
        self.assertTrue(shape_test and mean_test and std_test)

    def test_gaussian_white_noise_3d_vector_sigma(self) -> None:
        size = (100_000, 1_000, 3)
        sigma = [.25, .5, 1.]
        process = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        shape_test = process.shape == size
        mean_test = np.isclose(process.mean(), 0., atol=TOLERANCE)
        std_test = np.allclose(process.std(axis=0).mean(axis=0), np.array(sigma), atol=TOLERANCE)
        self.assertTrue(shape_test and mean_test and std_test)


class TestWiener(unittest.TestCase):

    def test_wiener_2d(self) -> None:
        periods = 12
        num_periods = 10
        size = (periods * num_periods, 500_000)
        process = Wiener(size=size, periods=periods).generate()
        shape_test = process.shape == (size[0] + 1, ) + size[1:]
        mean_test = np.isclose(process[-1, :].mean(), 0., atol=TOLERANCE)
        std_test = np.isclose(process[-1, :].std(axis=0).mean(), sqrt(num_periods), atol=TOLERANCE)
        self.assertTrue(shape_test and mean_test and std_test)

    def test_wiener_3d_no_correlation(self) -> None:
        periods = 12
        num_periods = 10
        num_assets = 3
        size = (periods * num_periods, 500_000, num_assets)
        process = Wiener(size=size, periods=periods).generate()
        shape_test = process.shape == (size[0] + 1, ) + size[1:]
        mean_test = np.isclose(process[-1, :].mean(), 0., atol=TOLERANCE)
        std_test = np.isclose(process[-1, :].std(axis=0).mean(), sqrt(num_periods), atol=TOLERANCE)
        correlations = np.corrcoef(process.reshape(-1, process.shape[-1]).T)
        corr_test = np.allclose(correlations, np.identity(n=num_assets), atol=TOLERANCE)
        self.assertTrue(shape_test and mean_test and std_test and corr_test)

    def test_wiener_3d_with_correlation(self) -> None:
        periods = 12
        num_periods = 10
        num_assets = 3
        size = (periods * num_periods, 500_000, num_assets)
        rho = np.array([[1., .25, .5], [.25, 1., .75], [.5, .75, 1.]])
        process = Wiener(size=size, rho=rho, periods=periods).generate()
        shape_test = process.shape == (size[0] + 1, ) + size[1:]
        mean_test = np.isclose(process[-1, :].mean(), 0., atol=TOLERANCE)
        std_test = np.allclose(process[-1, :].std(axis=0), sqrt(num_periods), atol=TOLERANCE)
        correlations = np.corrcoef(process.reshape(-1, process.shape[-1]).T)
        corr_test = np.allclose(correlations, rho, atol=TOLERANCE)
        self.assertTrue(shape_test and mean_test and std_test and corr_test)


if __name__ == '__main__':
    unittest.main()
