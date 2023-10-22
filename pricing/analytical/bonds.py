from abc import abstractmethod
from typing import Union

from pricing.tools.helper import Balance
from pricing.tools.utils import FinancialProduct


class Bond(FinancialProduct):
    """
    Generic abstract implementation of the Bond object. The only method shared for all kinds of pure bonds is the
    need to discount the present value of the principal repayment, so this class implements this method. The pricing
    method is defined as an abstract method to be implemented for each specific Bond depending on its features.
    """

    __slots__ = ['p', 't']

    def __init__(self, p: Union[int, float], t: Union[int, float]) -> None:
        """
        Method to initialize the Bond class with its relevant input.

        :param p: principal of the bond to be repaid at maturity.
        :param t: time to bond maturity in which the principal will be repaid.
        """
        object.__setattr__(self, 'p', p)
        object.__setattr__(self, 't', t)
        object.__setattr__(self, '_ignore', ['_ignore'])

    def get_present_value_principal_repayment(self, r: float, compounding: str) -> float:
        """
        Method to calculate the present value of the principal repayment considering the interest rate :math:`r`.
        The present value is calculated at :math:`t_0=T-t` where :math:`T` is the maturity date. The implementation
        relies on the Balance object and its discount method. Please, note that :math:`r` must be measured
        in the same time units as :math:`t`, i.e., if :math:`t` is measured in years, :math:`r` should be the
        annualized interest rate.

        :param r: interest rate used to discount future cash flows.
        :param compounding: type of compounding used, accepted values are discrete and continuous.
        :return: present value of the principal at time :math:`t_0` using interest rate :math:`r`.
        """
        pv_p = Balance(amount=self.p).discount(r=r, t=self.t, compounding=compounding)
        return pv_p

    @abstractmethod
    def price(self, r: float, compounding: str) -> float:
        """
        Method to calculate the Bond price. The implementation needs to be adjusted for the particularities
        of the specific Bond such as the coupon structure. Please, note that :math:`r` must be measured
        in the same time units as :math:`t`, i.e., if :math:`t` is measured in years, :math:`r` should be the
        annualized interest rate.

        :param r: interest rate used to discount future cash flows.
        :param compounding: type of compounding used, accepted values are discrete and continuous.
        :return: Price calculated for the Bond.
        """
        pass

    @abstractmethod
    def get_ytm(self, price: Union[int, float]) -> float:
        """
        Method to calculate the yield to maturity (YTM) of the Bond given the current market price. The implementation
        needs to be adjusted for the particularities of the specific Bond such as the coupon structure. Please, note
        that the timeframe of reference for the YTM will be that of :math:`t`, i.e., if :math:`t` is measured in years,
        the YTM will be an annualized interest rate.

        :param price: price for the zero-coupon Bond observed in the market.
        :return: YTM calculated for the Bond.
        """
        pass


class ZeroCouponBond(Bond):
    """
    Implementation of the zero-coupon Bond valuation by discounting the principal repayment. The zero-coupon bond
    is a kind of bond that does not bear coupons, only the principal is repaid at maturity.
    """

    def price(self, r: float, compounding: str) -> float:
        """
        Method to calculate the Bond price. In the case of the zero-coupon Bond, since there are no coupons, the price
        is obtained simply by discounting the principal repayment at the appropriate interest rate.

        :param r: interest rate used to discount future cash flows.
        :param compounding: type of compounding used, accepted values are discrete and continuous.
        :return: Price calculated for the zero-coupon Bond.
        """
        price = self.get_present_value_principal_repayment(r=r, compounding=compounding)
        return price

    def get_ytm(self, price: Union[int, float]) -> float:
        """
        Method to calculate the yield to maturity (YTM) of the zero-coupon Bond given the current market price.
        In the case of the zero-coupon Bond there is a straight-forward closed analytical expression. Please, note
        that the timeframe of reference for the YTM will be that of :math:`t`, i.e., if :math:`t` is measured in
        years, the YTM will be an annualized interest rate.

        :param price: price for the zero-coupon Bond observed in the market.
        :return: YTM calculated for the Bond.
        """
        ytm = (self.p / price) ** (1 / self.t) - 1
        return ytm
