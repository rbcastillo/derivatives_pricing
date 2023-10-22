from typing import Union

from pricing.analytical.bonds import ZeroCouponBond
from pricing.tools.helper import Balance
from pricing.tools.utils import FinancialProduct


class Forward(FinancialProduct):
    """
    Implementation of the Forward contract valuation by using the analytical expression. The Forward contract
    provides the holder the right and the obligation, to buy an underlying asset, at a predetermined forward price
    on a specified expiration date. Usually the forward price is fixed at the beginning of the contract in such way
    that no money is paid in advance for entering the agreement.
    """

    __slots__ = ['s', 'r', 't', 'q']

    def __init__(self, s: Union[float, int], r: float, t: Union[float, int], q: float = 0) -> None:
        """
        Method to initialize the Forward contract class with its relevant input. Please, note that the metrics r
        and q must be measured in the same time units as t, i.e., if t is measured in years, r and q should be
        annualized terms.

        :param s: current value for the underlying security.
        :param r: risk-free interest rate.
        :param t: time to maturity for the option contract.
        :param q: continuous rate of dividends paid by the underlying security.
        """
        object.__setattr__(self, 's', s)
        object.__setattr__(self, 'r', r)
        object.__setattr__(self, 't', t)
        object.__setattr__(self, 'q', q)

    def get_forward_price(self) -> float:
        """
        The equilibrium forward price for the contract so that no party is better off is:

        :math:`F_0 = S e^{(r-q)t}`

        Therefore, that is usually the forward price set at the writing of the contract.

        The calculation relies on the implementation of the :ref:`Balance <balance>` object
        and its capitalization method considering the underlying price as the balance amount and :math:`r-q` the
        applicable interest rate.

        :return: calculated equilibrium forward price.
        """
        f_0 = Balance(amount=self.s).capitalize(r=self.r - self.q, t=self.t, compounding='continuous')
        return f_0

    def price(self, f_0: Union[int, float]) -> float:
        """
        The price for a Forward contract with :math:`F_0` forward price is calculated as:

        :math:`F = S e^{-qt} - F_0 e^{-rt}`

        The calculation relies on the implementation of the :ref:`zero-coupon Bond <zero_coupon_bond>`. The price is
        modelled as a zero-coupon Bond with principal equal to the current value of the underlying and discount rate
        equal to the continuous dividend yield of the underlying minus another zero-coupon Bond with principal equal
        to the forward price and discount rate equal to the risk-free rate.

        :param f_0: forward price specified for the Forward contract.
        :return: current value of the Forward position.
        """
        spot_price = ZeroCouponBond(p=self.s, t=self.t).price(r=self.q, compounding='continuous')
        forward_price_pv = ZeroCouponBond(p=f_0, t=self.t).price(r=self.r, compounding='continuous')
        price = spot_price - forward_price_pv
        return price
