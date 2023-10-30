import unittest
import numpy as np

from pricing.analytical.european_options import EuropeanCall, EuropeanPut
from pricing.analytical.forwards import Forward


class TestPricing(unittest.TestCase):

    def test_call_pricing(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        call_price = call.price()
        self.assertEqual(round(call_price, 9), 14.828453692)

    def test_put_pricing(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        put_price = put.price()
        self.assertEqual(round(put_price, 9), 17.800805857)

    def test_put_call_parity(self) -> None:
        s, k, r, t, sigma, q = 100, 120, 0.05, 5, 0.2, 0.02
        call_price = EuropeanCall(s=s, k=k, r=r, t=t, sigma=sigma, q=q).price()
        put_price = EuropeanPut(s=s, k=k, r=r, t=t, sigma=sigma, q=q).price()
        fw_price = Forward(s=s, r=r, t=t, q=q).price(f_0=k)
        parity = np.isclose(call_price - put_price, fw_price)
        self.assertTrue(parity)


class TestGreeks(unittest.TestCase):

    def test_call_delta(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        delta = call.get_delta()
        self.assertEqual(round(delta, 9), 0.506838995)

    def test_call_gamma(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        gamma = call.get_gamma()
        self.assertEqual(round(gamma, 9), 0.00797981)

    def test_call_theta(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        theta = call.get_theta()
        self.assertEqual(round(theta, 9), -2.375056276)

    def test_call_vega(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        vega = call.get_vega()
        self.assertEqual(round(vega, 9), 79.798098816)

    def test_call_rho_r(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        rho = call.get_rho(target='r')
        self.assertEqual(round(rho, 9), 179.277228888)

    def test_call_rho_q(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        rho = call.get_rho(target='q')
        self.assertEqual(round(rho, 9), -173.575352886)

    def test_call_implied_vol(self) -> None:
        volatility = 0.2
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=volatility, q=0.02)
        call_price = call.price()
        implied_volatility = call.calc_implied_vol(call_price)
        self.assertEqual(round(implied_volatility, 9), volatility)

    def test_put_delta(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        delta = put.get_delta()
        self.assertEqual(round(delta, 9), -0.397998423)

    def test_put_gamma(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        gamma = put.get_gamma()
        self.assertEqual(round(gamma, 9), 0.00797981)

    def test_put_theta(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        theta = put.get_theta()
        self.assertEqual(round(theta, 9), 0.488073587)

    def test_put_vega(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        vega = put.get_vega()
        self.assertEqual(round(vega, 9), 79.798098816)

    def test_put_rho_r(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        rho = put.get_rho(target='r')
        self.assertEqual(round(rho, 9), -288.003240955)

    def test_put_rho_q(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        rho = put.get_rho(target='q')
        self.assertEqual(round(rho, 9), 278.843356132)

    def test_put_implied_vol(self) -> None:
        volatility = 0.2
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=volatility, q=0.02)
        put_price = put.price()
        implied_volatility = put.calc_implied_vol(put_price)
        self.assertEqual(round(implied_volatility, 9), volatility)


class TestSpecial(unittest.TestCase):

    def test_string_casting_call(self) -> None:
        parameters = {'s': 100, 'k': 120, 'r': 0.05, 't': 5, 'sigma': 0.2, 'q': 0.02}
        call = EuropeanCall(**parameters)
        expected = f'EuropeanCall object with parameters {parameters}'
        self.assertEqual(str(call), expected)

    def test_string_casting_put(self) -> None:
        parameters = {'s': 100, 'k': 120, 'r': 0.05, 't': 5, 'sigma': 0.2, 'q': 0.02}
        put = EuropeanPut(**parameters)
        expected = f'EuropeanPut object with parameters {parameters}'
        self.assertEqual(str(put), expected)

    def test_change_existing_param_generic_call(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        call.t = 10
        self.assertTrue(call.t == 10 and call._d1 is None)

    def test_change_existing_param_generic_put(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        put.t = 10
        self.assertTrue(put.t == 10 and put._d1 is None)

    def test_change_existing_param_method_call(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        call.update_params(t=10)
        self.assertTrue(call.t == 10 and call._d1 is None)

    def test_change_existing_param_method_put(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        put.update_params(t=10)
        self.assertTrue(put.t == 10 and put._d1 is None)

    def test_add_new_param_generic_call(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            call.not_existent = 5
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute name <not_existent> is not recognized, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_add_new_param_generic_put(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            put.not_existent = 5
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute name <not_existent> is not recognized, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_add_new_param_method_call(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            call.update_params(not_existent=5)
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute name <not_existent> is not recognized, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_add_new_param_method_put(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            put.update_params(not_existent=5)
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute name <not_existent> is not recognized, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_change_existing_param_forbidden_generic_call(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            call._d1 = 5
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute <_d1> is protected or private, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_change_existing_param_forbidden_generic_put(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            put._d1 = 5
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute <_d1> is protected or private, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_change_existing_param_forbidden_method_call(self) -> None:
        call = EuropeanCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            call.update_params(_d1=5)
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute <_d1> is protected or private, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_change_existing_param_forbidden_method_put(self) -> None:
        put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            put.update_params(_d1=5)
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute <_d1> is protected or private, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)


if __name__ == '__main__':
    unittest.main()
