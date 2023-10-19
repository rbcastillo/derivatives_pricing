import unittest
import numpy as np

from pricing.analytical.forwards import Forward


class TestPricing(unittest.TestCase):

    def test_initial_pricing(self):
        fw = Forward(s=100, r=0.05, t=1, q=0.02)
        fw_price = fw.price(f_0=120)
        self.assertEqual(round(fw_price, 9), -16.127663609)

    def test_forward_pricing(self):
        fw = Forward(s=100, r=0.05, t=1, q=0.02)
        f_0 = fw.get_forward_price()
        self.assertEqual(round(f_0, 9), 103.045453395)

    def test_full_pricing(self):
        s = np.random.uniform(1, 500)
        r = np.random.uniform(0.01, 0.1)
        t = np.random.uniform(0.5, 5)
        q = np.random.uniform(0.01, 0.05)

        fw = Forward(s=s, r=r, t=t, q=q)
        print(fw.price(fw.get_forward_price()))
        price_matching = np.isclose(fw.price(fw.get_forward_price()), 0)
        self.assertTrue(price_matching)


if __name__ == '__main__':
    unittest.main()
