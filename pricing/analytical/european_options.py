from abc import ABC, abstractmethod
from math import sqrt, exp, log
from scipy.stats import norm
from typing import Union


class EuropeanOption(ABC):
    """
    Generic abstract implementation of the European option object populated with the key auxiliary methods
    shared between Call and Put options. The pricing method is defined as an abstract method to be implemented
    in the Call and Put specific objects depending on their features.
    """

    def __init__(self, s: Union[float, int], k: Union[float, int], r: float, t: Union[float, int], sigma: float,
                 q: float = 0) -> None:
        """
        Method to initialize the European option class with its relevant input. Please, note that the metrics r, sigma
        and q must be measured in the same time units as t, i.e., if t is measured in years, r, sigma and q should be
        annualized terms.

        :param s: current value for the underlying security.
        :param k: strike price for the option contract.
        :param r: risk-free interest rate.
        :param t: time to maturity for the option contract.
        :param sigma: volatility level for the underlying security.
        :param q: continuous rate of dividends paid by the underlying security.
        """
        self._s = s
        self._k = k
        self._r = r
        self._t = t
        self._sigma = sigma
        self._q = q
        self.d1, self.d2 = None, None

    def __str__(self) -> str:
        attr_values = {attr[1:]: value for (attr, value) in self.__dict__.items() if attr.startswith('_')}
        text_output = f'{self.__class__.__name__} object with parameters {attr_values}'
        return text_output

    def update_params(self, **kwargs) -> None:
        """
        This method allows updating some or all of the relevant input for the European option class. The kwargs used
        in the method must match those in the __init__ method.

        :param kwargs: keyword parameter(s) and value(s) to be updated.
        :return: None.
        """
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f'Parameter name {param} is not recognized.')
        self.d1 = None
        self.d2 = None

    def get_d1(self) -> float:
        """
        Method to calculate the auxiliary measure d1 used in the Black-Scholes formula for European options. The value
        for this measure is the same in the case of both Call and Put options.

        The formula for :math:`d1` is the following:

        :math:`\\small \\displaystyle d_1 = \\frac{\\ln{\\frac{S}{K}} + (r - q + \\frac{\\sigma^2}{2}) t}
        {\\sigma \\sqrt{t}}`

        :return: calculated value for :math:`d1`.
        """
        if self.d1 is None:
            d1_numerator = log(self._s / self._k) + (self._r - self._q + (self._sigma ** 2) / 2) * self._t
            d1_denominator = self._sigma * sqrt(self._t)
            self.d1 = d1_numerator / d1_denominator
        return self.d1

    def get_d2(self) -> float:
        """
        Method to calculate the auxiliary measure d1 used in the Black-Scholes formula for European options. The value
        for this measure is the same in the case of both Call and Put options.

        The formula for :math:`d1` is the following:

        :math:`\\small \\displaystyle d_2 = \\frac{\\ln{\\frac{S}{K}} + (r - q - \\frac{\\sigma^2}{2}) t}
        {\\sigma \\sqrt{t}} = d_1 - \\sigma \\sqrt{t}`

        :return: calculated value for :math:`d2`.
        """
        if self.d2 is None:
            d1 = self.get_d1()
            self.d2 = d1 - self._sigma * sqrt(self._t)
        return self.d2

    @abstractmethod
    def price(self) -> float:
        """
        Method to calculate the European option price, the implementation needs to be adjusted for either Call or
        Put options in the specific classes.

        :return: Price calculated for the European option.
        """
        pass

    def get_gamma(self) -> float:
        """
        The gamma of a European option represents the expected change in the delta of the option as a response to
        small changes in the price of the underlying. It represents the convexity of the option value to changes in
        the underlying asset via the second order effect (:math:`\\small \\displaystyle \\frac{\\partial^2 V}
        {\\partial S^2}`). The gamma of the European Call option is calculated as:

        :math:`\\small \\displaystyle \\Gamma = \\frac{e^{-qt} N'(d_1)}{S \\sigma \\sqrt{t}}`

        Where :math:`N'(x)` represents the PDF of the standard Gaussian distribution and :math:`d_1` is defined in
        its own method.

        :return: Gamma calculated for the European option.
        """
        d1 = self.get_d1()
        gamma = exp(-self._q * self._t) * norm.pdf(d1) / (self._s * self._sigma * sqrt(self._t))
        return gamma


class EuropeanCall(EuropeanOption):
    """
    Implementation of the European Call option valuation by using the analytical expression. The European Call option
    provides the holder the right, but not the obligation, to buy an underlying asset, at a predetermined strike price
    on a specified expiration date.
    """

    def price(self) -> float:
        """
        The price of the European Call option is calculated as:

        :math:`\\small \\displaystyle C = S e^{-q t}N(d_1) - K e^{-r t}N(d_2)`

        Where :math:`N(x)` represents the CDF of the standard Gaussian distribution and :math:`d_1` and :math:`d_2`
        are defined in their respective methods.

        :return: Price calculated for the European Call option.
        """
        d1 = self.get_d1()
        d2 = self.get_d2()
        price = self._s * exp(-self._q * self._t) * norm.cdf(d1) - self._k * exp(-self._r * self._t) * norm.cdf(d2)
        return price

    def get_delta(self) -> float:
        """
        The delta of a European option represents the expected change in the price of the option as a response to
        small changes in the price of the underlying (:math:`\\small \\displaystyle \\frac{\\partial V}{\\partial S}`).
        The delta of the European Call option is calculated as:

        :math:`\\small \\displaystyle \\Delta = N(d_1)`

        Where :math:`N(x)` represents the CDF of the standard Gaussian distribution and :math:`d_1` is defined in
        its own method.

        :return: Delta calculated for the European Call option.
        """
        d1 = self.get_d1()
        delta = norm.cdf(d1) * exp(-self._q * self._t)
        return delta

    def get_theta(self, periods: int = 1) -> float:
        """
        The theta of a European option represents the expected change in the price of the option as a response to
        time passing (:math:`\\small \\displaystyle \\frac{\\partial V}{\\partial t}`). The theta of the European Call
        option is calculated as:

        :math:`\\small \\displaystyle \\Theta = - \\frac{\\sigma S e^{-qt} N'(d_1)}{2 \\sqrt{t}}
        + qSN(d_1)e^{-qt} - rKe^{-rt}N(d_2)`

        Where :math:`N(x)` and :math:`N'(x)` represents respectively the CDF and PDF of the standard Gaussian
        distribution and :math:`d_1` and :math:`d_2` are defined in their respective methods.

        As presented above, :math:`\\Theta` measures the decrease in value in the European option as one full t period
        passes by. Usually t is measured in years, so this measure is too large to be useful in usual European options
        with shorter maturities. For this reason, :math:`\\Theta` is usually divided by a :math:`T` factor representing
        the period of interest. For instance :math:`T=365` would measure the decrease in value for each calendar day,
        whereas :math:`T=252` would measure the decrease in value for each trading day.

        :param periods: period size to express theta, 365 for calendar days, 252 for trading days, etc.
        :return: Theta calculated for the European Call option.
        """
        d1 = self.get_d1()
        d2 = self.get_d2()
        first_element = (self._sigma * self._s * exp(-self._q * self._t) * norm.pdf(d1)) / (2 * sqrt(self._t))
        second_element = self._q * self._s * exp(-self._q * self._t) * norm.cdf(d1)
        third_element = self._r * self._k * exp(-self._r * self._t) * norm.cdf(d2)
        theta = (- first_element + second_element - third_element) / periods
        return theta


class EuropeanPut(EuropeanOption):
    """
    Implementation of the European Put option valuation by using the analytical expression. The European Put option
    provides the holder the right, but not the obligation, to sell an underlying asset, at a predetermined strike price
    on a specified expiration date.
    """

    def price(self) -> float:
        """
        The price of the European Put option is calculated as:

        :math:`\\small \\displaystyle P = K e^{-r t}N(-d_2) - S e^{-q t}N(-d_1)`

        Where :math:`N(x)` represents the CDF of the standard Gaussian distribution and :math:`d_1` and :math:`d_2`
        are defined in their respective methods.

        :return: Price calculated for the European Put option.
        """
        d1 = self.get_d1()
        d2 = self.get_d2()
        price = self._k * exp(-self._r * self._t) * norm.cdf(-d2) - self._s * exp(-self._q * self._t) * norm.cdf(-d1)
        return price

    def get_delta(self) -> float:
        """
        The delta of a European option represents the expected change in the price of the option as a response to
        small changes in the price of the underlying (:math:`\\small \\displaystyle \\frac{\\partial V}{\\partial S}`).
        The delta of the European Put option is calculated as:

        :math:`\\small \\displaystyle \\Delta = N(d_1) - 1`

        Where :math:`N(x)` represents the CDF of the standard Gaussian distribution and :math:`d_1` is defined in
        its own method.

        :return: Delta calculated for the European Put option.
        """
        d1 = self.get_d1()
        delta = (norm.cdf(d1) - 1) * exp(-self._q * self._t)
        return delta

    def get_theta(self, periods: int = 1) -> float:
        """
        The theta of a European option represents the expected change in the price of the option as a response to
        time passing (:math:`\\small \\displaystyle \\frac{\\partial V}{\\partial t}`). The theta of the European Put
        option is calculated as:

        :math:`\\small \\displaystyle \\Theta = - \\frac{\\sigma S e^{-qt} N'(d_1)}{2 \\sqrt{t}}
        - qSN(-d_1)e^{-qt} + rKe^{-rt}N(-d_2)`

        Where :math:`N(x)` and :math:`N'(x)` represents respectively the CDF and PDF of the standard Gaussian
        distribution and :math:`d_1` and :math:`d_2` are defined in their respective methods.

        As presented above, :math:`\\Theta` measures the decrease in value in the European option as one full t period
        passes by. Usually t is measured in years, so this measure is too large to be useful in usual European options
        with shorter maturities. For this reason, :math:`\\Theta` is usually divided by a :math:`T` factor representing
        the period of interest. For instance :math:`T=365` would measure the decrease in value for each calendar day,
        whereas :math:`T=252` would measure the decrease in value for each trading day.

        :param periods: period size to express theta, 365 for calendar days, 252 for trading days, etc.
        :return: Theta calculated for the European Put option.
        """
        d1 = self.get_d1()
        d2 = self.get_d2()
        first_element = (self._s * self._sigma * exp(-self._q * self._t) * norm.pdf(d1)) / (2 * sqrt(self._t))
        second_element = self._r * self._k * exp(-self._r * self._t) * norm.cdf(-d2)
        third_element = self._q * self._s * exp(-self._q * self._t) * norm.cdf(-d1)
        theta = (- first_element + second_element - third_element) / periods
        return theta


if __name__ == '__main__':
    call = EuropeanCall(s=100, k=120, r=0.05, t=1, sigma=0.2, q=0.02)
    print(call)
    print('Call:', round(call.price(), 9))
    print('Call delta:', round(call.get_delta(), 9))
    print('Call gamma:', round(call.get_gamma(), 9))
    print('Call theta:', round(call.get_theta(1), 9))

    print('\n')
    put = EuropeanPut(s=100, k=120, r=0.05, t=1, sigma=0.2, q=0.02)
    print('Put:', round(put.price(), 9))
    print('Put delta:', round(put.get_delta(), 9))
    print('Put gamma:', round(put.get_gamma(), 9))
    print('Put theta:', round(put.get_theta(1), 9))
