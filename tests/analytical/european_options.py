import unittest
import numpy as np

from pricing.analytical.european_options import EuropeanCall, EuropeanPut
from pricing.analytical.forwards import Forward


class TestPricing(unittest.TestCase):

    def test_call_pricing(self):
        call = EuropeanCall(s=100, k=120, r=0.05, t=1, sigma=0.2, q=0.02)
        call_price = call.price()
        self.assertEqual(round(call_price, 9), 2.711776128)

    def test_put_pricing(self):
        put = EuropeanPut(s=100, k=120, r=0.05, t=1, sigma=0.2, q=0.02)
        put_price = put.price()
        self.assertEqual(round(put_price, 9), 18.839439738)

    def test_put_call_parity(self):
        s = np.random.uniform(1, 500)
        k = np.random.uniform(max(1, s - 25), s + 25)
        r = np.random.uniform(0.01, 0.1)
        t = np.random.uniform(0.5, 5)
        sigma = np.random.uniform(0.01, 0.25)
        q = np.random.uniform(0.01, 0.05)

        call_price = EuropeanCall(s=s, k=k, r=r, t=t, sigma=sigma, q=q).price()
        put_price = EuropeanPut(s=s, k=k, r=r, t=t, sigma=sigma, q=q).price()
        fw_price = Forward(s=s, r=r, t=t, q=q).price(f_0=k)
        parity = np.isclose(call_price - put_price, fw_price)
        self.assertTrue(parity)


class TestGreeks(unittest.TestCase):

    def test_call_delta(self):
        call = EuropeanCall(s=100, k=120, r=0.05, t=1, sigma=0.2, q=0.02)
        call_price = call.price()
        self.assertEqual(round(call_price, 9), 2.711776128)


if __name__ == '__main__':
    unittest.main()
