import unittest
import numpy as np

from pricing.simulation.processes import GaussianWhiteNoise
from tests.env_variables import TOLERANCE


class TestGaussianWhiteNoise(unittest.TestCase):

    def test_gaussian_white_noise_1d(self) -> None:
        size = (500_000, )
        sigma = .5
        noise = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        shape_test = noise.shape == size
        mean_test = np.isclose(noise.mean(), 0., atol=TOLERANCE)
        std_test = np.isclose(noise.std(), sigma, atol=TOLERANCE)
        self.assertTrue(shape_test and mean_test and std_test)

    def test_gaussian_white_noise_2d(self) -> None:
        size = (100_000, 1_000)
        sigma = .5
        noise = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        shape_test = noise.shape == size
        mean_test = np.isclose(noise.mean(), 0., atol=TOLERANCE)
        std_test = np.isclose(noise.std(axis=0).mean(), sigma, atol=TOLERANCE)
        self.assertTrue(shape_test and mean_test and std_test)

    def test_gaussian_white_noise_3d_single_sigma(self) -> None:
        size = (100_000, 1_000, 3)
        sigma = .5
        noise = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        shape_test = noise.shape == size
        mean_test = np.isclose(noise.mean(), 0., atol=TOLERANCE)
        std_test = np.allclose(noise.std(axis=0).mean(axis=0), np.array([sigma] * size[2]), atol=TOLERANCE)
        self.assertTrue(shape_test and mean_test and std_test)

    def test_gaussian_white_noise_3d_vector_sigma(self) -> None:
        size = (100_000, 1_000, 3)
        sigma = [.25, .5, 1.]
        noise = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        shape_test = noise.shape == size
        mean_test = np.isclose(noise.mean(), 0., atol=TOLERANCE)
        std_test = np.allclose(noise.std(axis=0).mean(axis=0), np.array(sigma), atol=TOLERANCE)
        self.assertTrue(shape_test and mean_test and std_test)


if __name__ == '__main__':
    unittest.main()
