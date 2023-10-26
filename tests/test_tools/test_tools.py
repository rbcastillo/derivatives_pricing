import unittest
import numpy as np

from pricing.tools.helper import Balance
from pricing.tools.utils import StatisticalProcess


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


class TestStatisticalProcess(unittest.TestCase):

    def test_no_size(self):
        try:
            StatisticalProcess._manage_size(size=())
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError(f'Size <()> not valid, at least should have one dimension')
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_size_too_long(self):
        try:
            StatisticalProcess._manage_size(size=(1, 2, 3, 4))
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError(f'Size <(1, 2, 3, 4)> not valid, maximum number of dimensions is 3')
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_size_with_inner_zero(self):
        try:
            StatisticalProcess._manage_size(size=(10, 0, 10))
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError(f'Size <(10, 0, 10)> not valid, zero length inner dimensions are not allowed')
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_size_with_single_zero_end(self):
        size = StatisticalProcess._manage_size(size=(10, 10, 0))
        self.assertTrue(size == (10, 10))

    def test_size_with_double_zero_end(self):
        size = StatisticalProcess._manage_size(size=(10, 0, 0))
        self.assertTrue(size == (10, ))

    def test_size_with_all_zeros(self):
        try:
            StatisticalProcess._manage_size(size=(0, 0, 0))
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError(f'Size <()> not valid, at least should have one dimension')
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)


if __name__ == '__main__':
    unittest.main()
