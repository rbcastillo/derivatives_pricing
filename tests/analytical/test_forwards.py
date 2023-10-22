import unittest
import numpy as np

from pricing.analytical.forwards import Forward


class TestPricing(unittest.TestCase):

    def test_initial_pricing(self):
        fw = Forward(s=100, r=0.05, t=5, q=0.02)
        fw_price = fw.price(f_0=120)
        self.assertEqual(round(fw_price, 9), -2.972352165)

    def test_forward_pricing(self):
        fw = Forward(s=100, r=0.05, t=5, q=0.02)
        f_0 = fw.get_forward_price()
        self.assertEqual(round(f_0, 9), 116.183424273)

    def test_full_pricing(self):
        s, r, t, q = 100, 0.05, 5, 0.02
        fw = Forward(s=s, r=r, t=t, q=q)
        price_matching = np.isclose(fw.price(fw.get_forward_price()), 0)
        self.assertTrue(price_matching)


class TestSpecialForwards(unittest.TestCase):

    def test_string_casting(self):
        fw = Forward(s=100, r=0.05, t=5, q=0.02)
        expected = "Forward object with parameters {'s': 100, 'r': 0.05, 't': 5, 'q': 0.02}"
        self.assertEqual(str(fw), expected)

    def test_change_existing_param_generic(self):
        fw = Forward(s=100, r=0.05, t=5, q=0.02)
        fw.t = 10
        self.assertTrue(fw.t == 10)

    def test_change_existing_param_method(self):
        fw = Forward(s=100, r=0.05, t=5, q=0.02)
        fw.update_params(t=10)
        self.assertTrue(fw.t == 10)

    def test_add_new_param_generic(self):
        fw = Forward(s=100, r=0.05, t=5, q=0.02)
        try:
            fw.not_existent = 5
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute name <not_existent> is not recognized, "
                                  "use values in ['s', 'r', 't', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_add_new_param_method(self):
        fw = Forward(s=100, r=0.05, t=5, q=0.02)
        try:
            fw.update_params(not_existent=5)
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError("Attribute name <not_existent> is not recognized, "
                                  "use values in ['s', 'r', 't', 'q']")
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)


if __name__ == '__main__':
    unittest.main()
