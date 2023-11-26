import unittest
import numpy as np

from pricing.analytical.binary_options import BinaryCall, BinaryPut
from pricing.analytical.bonds import ZeroCouponBond


class TestPricing(unittest.TestCase):

    def test_call_pricing(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        call_price = call.price()
        self.assertEqual(round(call_price, 9), 0.298795381)

    def test_put_pricing(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        put_price = put.price()
        self.assertEqual(round(put_price, 9), 0.480005402)

    def test_put_call_parity(self) -> None:
        s, k, r, t, sigma, q = 100, 120, 0.05, 5, 0.2, 0.02
        call_price = BinaryCall(s=s, k=k, r=r, t=t, sigma=sigma, q=q).price()
        put_price = BinaryPut(s=s, k=k, r=r, t=t, sigma=sigma, q=q).price()
        payoff_pv = ZeroCouponBond(p=1., t=t).price(r=r, compounding='continuous')
        parity = np.isclose(call_price + put_price, payoff_pv)
        self.assertTrue(parity)


class TestGreeks(unittest.TestCase):

    def test_call_delta(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        delta = call.get_delta()
        self.assertEqual(round(delta, 9), 0.006649842)

    def test_call_gamma(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        gamma = call.get_gamma()
        self.assertEqual(round(gamma, 9), 0.00797981)

    def test_call_theta(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        theta = call.get_theta()
        self.assertEqual(round(theta, 9), -2.375056276)

    def test_call_vega(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        vega = call.get_vega()
        self.assertEqual(round(vega, 9), 79.798098816)

    def test_call_rho_r(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        rho = call.get_rho(target='r')
        self.assertEqual(round(rho, 9), 179.277228888)

    def test_call_rho_q(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        rho = call.get_rho(target='q')
        self.assertEqual(round(rho, 9), -173.575352886)

    def test_call_implied_vol(self) -> None:
        volatility = 0.2
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=volatility, q=0.02)
        call_price = call.price()
        implied_volatility = call.calc_implied_vol(call_price)
        self.assertEqual(round(implied_volatility, 9), volatility)

    def test_put_delta(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        delta = put.get_delta()
        self.assertEqual(round(delta, 9), -0.006649842)

    def test_put_gamma(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        gamma = put.get_gamma()
        self.assertEqual(round(gamma, 9), 0.00797981)

    def test_put_theta(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        theta = put.get_theta()
        self.assertEqual(round(theta, 9), 0.488073587)

    def test_put_vega(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        vega = put.get_vega()
        self.assertEqual(round(vega, 9), 79.798098816)

    def test_put_rho_r(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        rho = put.get_rho(target='r')
        self.assertEqual(round(rho, 9), -288.003240955)

    def test_put_rho_q(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        rho = put.get_rho(target='q')
        self.assertEqual(round(rho, 9), 278.843356132)

    def test_put_implied_vol(self) -> None:
        volatility = 0.2
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=volatility, q=0.02)
        put_price = put.price()
        implied_volatility = put.calc_implied_vol(put_price)
        self.assertEqual(round(implied_volatility, 9), volatility)


class TestImplementation(unittest.TestCase):

    def test_string_casting_call(self) -> None:
        parameters = {'s': 100, 'k': 120, 'r': 0.05, 't': 5, 'sigma': 0.2, 'q': 0.02}
        call = BinaryCall(**parameters)
        expected = f'BinaryCall object with parameters {parameters}'
        self.assertEqual(str(call), expected)

    def test_string_casting_put(self) -> None:
        parameters = {'s': 100, 'k': 120, 'r': 0.05, 't': 5, 'sigma': 0.2, 'q': 0.02}
        put = BinaryPut(**parameters)
        expected = f'BinaryPut object with parameters {parameters}'
        self.assertEqual(str(put), expected)

    def test_change_existing_param_generic_call(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        call.t = 10
        self.assertTrue(call.t == 10 and call._d1 is None)

    def test_change_existing_param_generic_put(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        put.t = 10
        self.assertTrue(put.t == 10 and put._d1 is None)

    def test_change_existing_param_method_call(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        call.update_params(t=10)
        self.assertTrue(call.t == 10 and call._d1 is None)

    def test_change_existing_param_method_put(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        put.update_params(t=10)
        self.assertTrue(put.t == 10 and put._d1 is None)

    def test_add_new_param_generic_call(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            call.not_existent = 5
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute name <not_existent> is not recognized, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_add_new_param_generic_put(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            put.not_existent = 5
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute name <not_existent> is not recognized, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_add_new_param_method_call(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            call.update_params(not_existent=5)
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute name <not_existent> is not recognized, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_add_new_param_method_put(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            put.update_params(not_existent=5)
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute name <not_existent> is not recognized, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_change_existing_param_forbidden_generic_call(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            call._d1 = 5
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute <_d1> is protected or private, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_change_existing_param_forbidden_generic_put(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            put._d1 = 5
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute <_d1> is protected or private, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_change_existing_param_forbidden_method_call(self) -> None:
        call = BinaryCall(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            call.update_params(_d1=5)
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute <_d1> is protected or private, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_change_existing_param_forbidden_method_put(self) -> None:
        put = BinaryPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
        try:
            put.update_params(_d1=5)
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute <_d1> is protected or private, "
                                  "use values in ['s', 'k', 'r', 't', 'sigma', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)


if __name__ == '__main__':
    unittest.main()
