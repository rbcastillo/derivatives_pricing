import unittest
import numpy as np

from pricing.analytical.bonds import ZeroCouponBond


class TestPricingZeroCoupon(unittest.TestCase):

    def test_pricing_discrete(self) -> None:
        price = ZeroCouponBond(p=1_000, t=5).price(r=0.05, compounding='discrete')
        self.assertEqual(round(price, 9), 783.526166468)

    def test_pricing_continuous(self) -> None:
        price = ZeroCouponBond(p=1_000, t=5).price(r=0.05, compounding='continuous')
        self.assertEqual(round(price, 9), 778.800783071)


class TestYTMZeroCoupon(unittest.TestCase):

    def test_ytm(self) -> None:
        ytm = ZeroCouponBond(p=1_000, t=5).get_ytm(price=800)
        self.assertEqual(round(ytm, 9), 0.045639553)

    def test_ytm_matching(self) -> None:
        irr = 0.05
        theoretical_price = ZeroCouponBond(p=1_000, t=5).price(r=irr, compounding='discrete')
        ytm = ZeroCouponBond(p=1_000, t=5).get_ytm(price=theoretical_price)
        ytm_matching = np.isclose(ytm, irr)
        self.assertTrue(ytm_matching)


class TestImplementationZeroCoupon(unittest.TestCase):

    def test_string_casting(self) -> None:
        zc_bond = ZeroCouponBond(p=1_000, t=5)
        expected = "ZeroCouponBond object with parameters {'p': 1000, 't': 5}"
        self.assertEqual(str(zc_bond), expected)

    def test_change_existing_param_generic(self) -> None:
        zc_bond = ZeroCouponBond(p=1_000, t=5)
        zc_bond.t = 10
        self.assertTrue(zc_bond.t == 10)

    def test_change_existing_param_method(self) -> None:
        zc_bond = ZeroCouponBond(p=1_000, t=5)
        zc_bond.update_params(t=10)
        self.assertTrue(zc_bond.t == 10)

    def test_add_new_param_generic(self) -> None:
        zc_bond = ZeroCouponBond(p=1_000, t=5)
        try:
            zc_bond.not_existent = 5
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute name <not_existent> is not recognized, use values in ['p', 't']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_add_new_param_method(self) -> None:
        zc_bond = ZeroCouponBond(p=1_000, t=5)
        try:
            zc_bond.update_params(not_existent=5)
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute name <not_existent> is not recognized, use values in ['p', 't']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)


if __name__ == '__main__':
    unittest.main()
