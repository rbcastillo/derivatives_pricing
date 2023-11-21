import unittest

import numpy as np

from tools.processes import GaussianWhiteNoise, Wiener, GeometricBrownianMotion
from tests.test_tools.helper import HelperGaussianWhiteNoise, HelperWiener, HelperGBM


class TestGaussianWhiteNoise(unittest.TestCase):

    def test_gaussian_white_noise_2d(self) -> None:
        np.random.seed(42)
        size, sigma = (1_000, 100_000), .5
        process = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        self.assertTrue(all(HelperGaussianWhiteNoise.calc_tests(process, size, mu=0, sigma=sigma)))

    def test_gaussian_white_noise_3d_single_sigma(self) -> None:
        np.random.seed(42)
        size, sigma = (250, 100_000, 3), .5
        process = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        self.assertTrue(all(HelperGaussianWhiteNoise.calc_tests(process, size, mu=0, sigma=sigma)))

    def test_gaussian_white_noise_3d_vector_sigma(self) -> None:
        np.random.seed(42)
        size, sigma = (250, 100_000, 3), [.25, .5, 1.]
        process = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        self.assertTrue(all(HelperGaussianWhiteNoise.calc_tests(process, size, mu=0, sigma=sigma)))

    def test_consistency(self) -> None:
        np.random.seed(42)
        size, sigma = (250, 100_000, 3), [.25, .5, 1.]
        process = GaussianWhiteNoise(size=size, sigma=sigma).generate()
        self.assertEqual(process[-1].mean().round(9), 0.002093015)


class TestWiener(unittest.TestCase):

    def test_wiener_2d(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths = 4, 12, 100_000
        size = (time_units * sub_periods, num_paths)
        w = Wiener(size=(time_units * sub_periods, 250), sub_periods=sub_periods)
        process = Wiener(size=size, sub_periods=sub_periods).generate()
        self.assertTrue(all(HelperWiener.calc_tests(w, process, size, time_units, 250)))

    def test_wiener_3d_no_correlation(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths, num_assets = 4, 12, 100_000, 3
        size = (time_units * sub_periods, num_paths, num_assets)
        w = Wiener(size=(time_units * sub_periods, 250, num_assets), sub_periods=sub_periods)
        process = Wiener(size=size, sub_periods=sub_periods).generate()
        self.assertTrue(all(HelperWiener.calc_tests(w, process, size, time_units, num_paths=250)))

    def test_wiener_3d_with_correlation(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths, num_assets = 4, 12, 100_000, 3
        size = (time_units * sub_periods, num_paths, num_assets)
        rho = np.array([[1., .25, .5], [.25, 1., .75], [.5, .75, 1.]])
        w = Wiener(size=(time_units * sub_periods, 250, num_assets), rho=rho, sub_periods=sub_periods)
        process = Wiener(size=size, rho=rho, sub_periods=sub_periods).generate()
        self.assertTrue(all(HelperWiener.calc_tests(w, process, size, time_units, num_paths=250, rho=rho)))

    def test_wiener_distribution_2d(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths = 4, 12, 500_000
        size = (time_units * sub_periods, num_paths)
        distribution = Wiener(size=size, rho=None, sub_periods=sub_periods).generate_distribution()
        self.assertTrue(all(HelperWiener.calc_distribution_tests(distribution, mu=0, sigma=np.sqrt(time_units))))

    def test_wiener_distribution_3d_with_correlation(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths, num_assets = 4, 12, 500_000, 3
        size = (time_units * sub_periods, num_paths, num_assets)
        rho = np.array([[1., .25, .5], [.25, 1., .75], [.5, .75, 1.]])
        distribution = Wiener(size=size, rho=rho, sub_periods=sub_periods).generate_distribution()
        self.assertTrue(all(HelperWiener.calc_distribution_tests(distribution, mu=0, sigma=np.sqrt(time_units))))

    def test_consistency(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths, num_assets = 4, 12, 100_000, 3
        size = (time_units * sub_periods, num_paths, num_assets)
        rho = np.array([[1., .25, .5], [.25, 1., .75], [.5, .75, 1.]])
        process = Wiener(size=size, rho=rho, sub_periods=sub_periods).generate()
        self.assertEqual(process[-1].mean().round(9), -0.000743865)


class TestGeometricBrownianMotion(unittest.TestCase):

    def test_gbm_2d(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths = 4, 12, 100_000
        size = (time_units * sub_periods, num_paths)
        params = {'s0': 100, 'mu': .15, 'sigma': .20, 'q': .05}

        gbm = GeometricBrownianMotion(size=(time_units * sub_periods, 250), **params, sub_periods=sub_periods)
        process = GeometricBrownianMotion(size=size, **params, sub_periods=sub_periods).generate()
        self.assertTrue(all(HelperGBM.calc_tests(gbm, process, size, time_units, 250, **params)))

    def test_gbm_distribution_2d(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths = 4, 12, 500_000
        size = (time_units * sub_periods, num_paths)
        params = {'s0': 100, 'mu': .15, 'sigma': .20, 'q': .05}

        distribution = GeometricBrownianMotion(size, **params, sub_periods=sub_periods).generate_distribution()
        self.assertTrue(all(HelperGBM.calc_distribution_tests(distribution, time_units, **params)))

    def test_gbm_3d_no_correlation(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths, n_assets = 4, 12, 100_000, 3
        size = (time_units * sub_periods, num_paths, n_assets)
        params = {'s0': 100, 'mu': .15, 'sigma': .20, 'q': .05}

        gbm = GeometricBrownianMotion(size=(time_units * sub_periods, 250, n_assets), **params, sub_periods=sub_periods)
        process = GeometricBrownianMotion(size=size, **params, sub_periods=sub_periods).generate()
        self.assertTrue(all(HelperGBM.calc_tests(gbm, process, size, time_units, 250, **params)))

    def test_gbm_3d_with_correlation(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths, n_assets = 4, 12, 100_000, 3
        size = (time_units * sub_periods, num_paths, n_assets)
        params = {'s0': 100, 'mu': .15, 'sigma': .20, 'q': .05,
                  'rho': np.array([[1., .25, .5], [.25, 1., .75], [.5, .75, 1.]])}

        gbm = GeometricBrownianMotion(size=(time_units * sub_periods, 250, n_assets), **params, sub_periods=sub_periods)
        process = GeometricBrownianMotion(size=size, **params, sub_periods=sub_periods).generate()
        self.assertTrue(all(HelperGBM.calc_tests(gbm, process, size, time_units, 250, **params)))

    def test_gbm_3d_with_correlation_vector_variables(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths, n_assets = 4, 12, 100_000, 3
        size = (time_units * sub_periods, num_paths, n_assets)
        params = {'s0': [100, 150, 50], 'mu': [.15, .10, .05], 'sigma': [.20, .10, .15], 'q': [.05, .02, .01],
                  'rho': np.array([[1., .25, .5], [.25, 1., .75], [.5, .75, 1.]])}

        gbm = GeometricBrownianMotion(size=(time_units * sub_periods, 250, n_assets), **params, sub_periods=sub_periods)
        process = GeometricBrownianMotion(size=size, **params, sub_periods=sub_periods).generate()
        self.assertTrue(all(HelperGBM.calc_tests(gbm, process, size, time_units, 250, **params)))

    def test_gbm_distribution_3d_with_correlation_vector_variables(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths, n_assets = 4, 12, 500_000, 3
        size = (time_units * sub_periods, num_paths, n_assets)
        params = {'s0': [100, 150, 50], 'mu': [.15, .10, .05], 'sigma': [.20, .10, .15], 'q': [.05, .02, .01]}
        rho = np.array([[1., .25, .5], [.25, 1., .75], [.5, .75, 1.]])

        distribution = GeometricBrownianMotion(size, **params, rho=rho, sub_periods=sub_periods).generate_distribution()
        self.assertTrue(all(HelperGBM.calc_distribution_tests(distribution, time_units, **params)))

    def test_consistency(self) -> None:
        np.random.seed(42)
        time_units, sub_periods, num_paths, n_assets = 4, 12, 100_000, 3
        size = (time_units * sub_periods, num_paths, n_assets)
        params = {'s0': [100, 150, 50], 'mu': [.15, .10, .05], 'sigma': [.20, .10, .15], 'q': [.05, .02, .01],
                  'rho': np.array([[1., .25, .5], [.25, 1., .75], [.5, .75, 1.]])}
        process = GeometricBrownianMotion(size=size, **params, sub_periods=sub_periods).generate()
        self.assertEqual(process[-1].mean().round(9), 138.147905671)


if __name__ == '__main__':
    unittest.main()
