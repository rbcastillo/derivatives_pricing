from math import exp
from typing import Union


class Forward:
    """
    Implementation of the Forward contract valuation by using the analytical expression. The Forward contract
    provides the holder the right and the obligation, to buy an underlying asset, at a predetermined forward price
    on a specified expiration date. Usually the forward price is fixed at the beginning of the contract in such way
    that no money is paid in advance for entering the agreement.
    """

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
        self.s = s
        self.r = r
        self.t = t
        self.q = q

    def __str__(self) -> str:
        attr_values = {attr[1:]: value for (attr, value) in self.__dict__.items() if attr.startswith('_')}
        text_output = f'{self.__class__.__name__} object with parameters {attr_values}'
        return text_output

    def get_forward_price(self) -> float:
        """
        The equilibrium forward price for the contract so that no party is better off is:

        :math:`F_0 = S e^{(r-q)t}`

        Therefore, that is usually the forward price set at the writing of the contract.

        :return: calculated equilibrium forward price.
        """
        f_0 = self.s * exp((self.r - self.q) * self.t)
        return f_0

    def price(self, f_0: Union[int, float]) -> float:
        """
        The price for a Forward contract with :math:`F_0` forward price is calculated as:

        :math:`F = S e^{-qt} - F_0 e^{-rt}`

        :param f_0: forward price specified for the Forward contract.
        :return: current value of the Forward position.
        """
        price = self.s * exp(-self.q * self.t) - f_0 * exp(-self.r * self.t)
        return price
