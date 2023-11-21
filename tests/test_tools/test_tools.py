import unittest
import numpy as np

from tools.helper import Balance
from tools.processes import StatisticalProcess


class TestBalance(unittest.TestCase):

    def test_capitalize_discrete(self) -> None:
        amount_fv = Balance(amount=100).capitalize(r=0.05, t=5, compounding='discrete')
        self.assertEqual(round(amount_fv, 9), 127.62815625)

    def test_capitalize_continuous(self) -> None:
        amount_fv = Balance(amount=100).capitalize(r=0.05, t=5, compounding='continuous')
        self.assertEqual(round(amount_fv, 9), 128.402541669)

    def test_discount_discrete(self) -> None:
        amount_fv = Balance(amount=100).discount(r=0.05, t=5, compounding='discrete')
        self.assertEqual(round(amount_fv, 9), 78.352616647)

    def test_discount_continuous(self) -> None:
        amount_fv = Balance(amount=100).discount(r=0.05, t=5, compounding='continuous')
        self.assertEqual(round(amount_fv, 9), 77.880078307)

    def test_capitalize_and_discount_discrete(self) -> None:
        start_value, r, t = 100, 0.05, 5
        compounding = 'discrete'
        account = Balance(amount=start_value)
        future_value = account.capitalize(r=r, t=t, compounding=compounding)
        discounted_future_value = Balance(amount=future_value).discount(r=r, t=t, compounding=compounding)
        balance_matching = np.isclose(discounted_future_value, start_value)
        self.assertTrue(balance_matching)

    def test_capitalize_and_discount_continuous(self) -> None:
        start_value, r, t = 100, 0.05, 5
        compounding = 'continuous'
        account = Balance(amount=start_value)
        future_value = account.capitalize(r=r, t=t, compounding=compounding)
        discounted_future_value = Balance(amount=future_value).discount(r=r, t=t, compounding=compounding)
        balance_matching = np.isclose(discounted_future_value, start_value)
        self.assertTrue(balance_matching)


class TestStatisticalProcess(unittest.TestCase):

    def test_no_size(self) -> None:
        try:
            StatisticalProcess._manage_size(size=())
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError('Size <()> not valid, at least should have one dimension')
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_size_too_long(self) -> None:
        try:
            StatisticalProcess._manage_size(size=(1, 2, 3, 4))
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError('Size <(1, 2, 3, 4)> not valid, maximum number of dimensions is 3')
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_size_with_inner_zero(self) -> None:
        try:
            StatisticalProcess._manage_size(size=(10, 0, 10))
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError('Size <(10, 0, 10)> not valid, zero length inner dimensions are not allowed')
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_size_with_single_zero_end(self) -> None:
        size = StatisticalProcess._manage_size(size=(10, 10, 0))
        self.assertTrue(size == (10, 10))

    def test_size_with_double_zero_end(self) -> None:
        size = StatisticalProcess._manage_size(size=(10, 0, 0))
        self.assertTrue(size == (10, ))

    def test_size_with_all_zeros(self) -> None:
        try:
            StatisticalProcess._manage_size(size=(0, 0, 0))
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError('Size <()> not valid, at least should have one dimension')
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_valid_parameter_asset_dimension_single_value(self) -> None:
        StatisticalProcess._check_parameter_with_asset_dimension(name='test', value=1, size=(10, 10, 2))

    def test_valid_parameter_asset_dimension_collection(self) -> None:
        StatisticalProcess._check_parameter_with_asset_dimension(name='test', value=[1, 2.5], size=(10, 10, 2))

    def test_parameter_asset_collection_with_no_asset_dimension(self) -> None:
        try:
            StatisticalProcess._check_parameter_with_asset_dimension(name='test', value=[1, 2.5], size=(10, 10))
            self.assertTrue(False)
        except ValueError as error:
            expected = ValueError('<test> is a collection but there is no asset dimension in size: <(10, 10)>')
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_parameter_asset_collection_with_not_matching_length(self) -> None:
        try:
            StatisticalProcess._check_parameter_with_asset_dimension(name='test', value=[1., 2.5], size=(10, 10, 5))
            self.assertTrue(False)
        except ValueError as error:
            error_msg = 'Length of <test> <[1.0, 2.5]> does not match the asset dimension in size: <(10, 10, 5)>'
            expected = ValueError(error_msg)
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_adjust_valid_parameter_asset_dimension_single_value(self) -> None:
        value = StatisticalProcess._adjust_parameter_with_asset_dimension(value=1., size=(10, 10, 2))
        self.assertTrue(np.array_equal(value, np.array([1, 1])))

    def test_adjust_valid_parameter_asset_dimension_collection(self) -> None:
        value = StatisticalProcess._adjust_parameter_with_asset_dimension(value=[1., 2.5], size=(10, 10, 2))
        self.assertTrue(np.array_equal(value, np.array([1., 2.5])))

    def test_valid_rho_matrix(self) -> None:
        rho = np.array([[10., 1., 5.], [1., 10., 5.], [1., 5., 10.]])
        rho = np.dot(rho, rho.transpose())
        rho /= np.max(rho)
        StatisticalProcess._check_rho_properties(rho=rho, n_assets=3)

    def test_rho_matrix_invalid_shape(self) -> None:
        try:
            rho = np.array([[1., .5], [.5, 1.]])
            StatisticalProcess._check_rho_properties(rho=rho, n_assets=3)
            self.assertTrue(False)
        except AssertionError as error:
            expected = AssertionError('rho shape <(2, 2)> does not match n_assets: 3')
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_rho_matrix_not_symmetric(self) -> None:
        try:
            rho = np.array([[1., .25, .5], [.25, 1., .75], [1., .25, 1.]])
            StatisticalProcess._check_rho_properties(rho=rho, n_assets=3)
            self.assertTrue(False)
        except AssertionError as error:
            expected = AssertionError('rho is not symmetric, not a valid correlation matrix')
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_rho_matrix_not_positive_definite(self) -> None:
        try:
            rho = np.array([[1., .25, .5], [.25, 1., -.75], [.5, -.75, 1.]])
            StatisticalProcess._check_rho_properties(rho=rho, n_assets=3)
            self.assertTrue(False)
        except AssertionError as error:
            expected = AssertionError('rho is not positive definite, not a valid correlation matrix')
            print(error)
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)

    def test_rho_matrix_invalid_values(self) -> None:
        try:
            rho = np.array([[2., .5, 1.], [.5, 2., 1.5], [1., 1.5, 2.]])
            StatisticalProcess._check_rho_properties(rho=rho, n_assets=3)
            self.assertTrue(False)
        except AssertionError as error:
            expected = AssertionError('rho values outside [-1, 1], not a valid correlation matrix')
            print(error)
            self.assertTrue(type(error) is type(expected) and error.args == expected.args)


if __name__ == '__main__':
    unittest.main()
