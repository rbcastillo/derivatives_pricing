import unittest
import numpy as np

from pricing.tools.helper import Balance


class TestBalance(unittest.TestCase):

    def test_capitalize_discrete(self):
        amount_fv = Balance(amount=100).capitalize(r=0.05, t=5, compounding='discrete')
        self.assertEqual(round(amount_fv, 9), 127.62815625)

    def test_capitalize_continuous(self):
        amount_fv = Balance(amount=100).capitalize(r=0.05, t=5, compounding='continuous')
        self.assertEqual(round(amount_fv, 9), 128.402541669)

    def test_discount_discrete(self):
        amount_fv = Balance(amount=100).discount(r=0.05, t=5, compounding='discrete')
        self.assertEqual(round(amount_fv, 9), 78.352616647)

    def test_discount_continuous(self):
        amount_fv = Balance(amount=100).discount(r=0.05, t=5, compounding='continuous')
        self.assertEqual(round(amount_fv, 9), 77.880078307)

    def test_capitalize_and_discount_discrete(self):
        start_value, r, t = 100, 0.05, 5
        compounding = 'discrete'
        account = Balance(amount=start_value)
        future_value = account.capitalize(r=r, t=t, compounding=compounding)
        discounted_future_value = Balance(amount=future_value).discount(r=r, t=t, compounding=compounding)
        balance_matching = np.isclose(discounted_future_value, start_value)
        self.assertTrue(balance_matching)

    def test_capitalize_and_discount_continuous(self):
        start_value, r, t = 100, 0.05, 5
        compounding = 'continuous'
        account = Balance(amount=start_value)
        future_value = account.capitalize(r=r, t=t, compounding=compounding)
        discounted_future_value = Balance(amount=future_value).discount(r=r, t=t, compounding=compounding)
        balance_matching = np.isclose(discounted_future_value, start_value)
        self.assertTrue(balance_matching)


if __name__ == '__main__':
    unittest.main()
