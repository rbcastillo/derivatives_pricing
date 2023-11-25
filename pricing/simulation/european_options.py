import numpy as np

from abc import abstractmethod
from math import sqrt, exp, log
from scipy.optimize import fmin
from scipy.stats import norm
from typing import Union, Any

from pricing.base import FinancialProduct
from tools.processes import StatisticalProcess


class EuropeanOption(FinancialProduct):
    """
    Generic abstract implementation of the European option object populated with the key auxiliary methods
    shared between Call and Put options. The pricing method is defined as an abstract method to be implemented
    in the Call and Put specific objects depending on their features.
    """

    __slots__ = ['price_process', 'k', 'r', 't', 's_t', 's_maturity']

    def __init__(self, price_process: StatisticalProcess, k: Union[float, int], r: float,
                 t: Union[float, int]) -> None:
        """
        Method to initialize the European option class with its relevant input. Please, note that the metrics
        :math:`r`, :math:`\\sigma` and :math:`q` must be measured in the same time units as :math:`t`, i.e., if
        :math:`t` is measured in years, :math:`r`, :math:`\\sigma` and :math:`q` should be annualized terms.

        :param price_process: statistical process object modelling the underlying price dynamics.
        :param k: strike price for the option contract.
        :param r: risk-free interest rate.
        :param t: time to maturity for the option contract.
        """
        object.__setattr__(self, 'price_process', price_process)
        object.__setattr__(self, 'k', k)
        object.__setattr__(self, 'r', r)
        object.__setattr__(self, 't', t)
        object.__setattr__(self, 's_t', None)
        object.__setattr__(self, 's_maturity', None)

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Method to manage the process of setting and updating object parameters. Only attributes declared in the
        __init__ method that are not protected or private are allowed to be updated. This avoids adding
        new attributes or modifying attributes that should only be modified inside the object as a result of
        certain operations.

        :param key: parameter to be added or updated.
        :param value: value to be assigned to the parameter.
        :return: None
        """
        super().__setattr__(key, value)
        # if key not in ['_d1', '_d2']:
        #     object.__setattr__(self, '_d1', None)
        #     object.__setattr__(self, '_d2', None)

    def get_process_simulation(self) -> np.ndarray:
        if self.s_t is None:
            object.__setattr__(self, 's_t', self.price_process.generate())
        return self.s_t

    def _calculate_process_ending_distribution(self, num_points: int = 500_000) -> np.ndarray:
        if self.s_maturity is None:
            s_maturity = self.price_process.generate_ending_distribution(num_points=num_points)
            object.__setattr__(self, 's_maturity', s_maturity)
        return self.s_maturity

    def _calculate_process_ending_simulation(self) -> np.ndarray:
        if self.s_maturity is None:
            s_t = self.get_process_simulation()
            object.__setattr__(self, 's_maturity', s_t[-1])
        return self.s_maturity

    def get_underlying_at_maturity(self, num_points: int = 500_000) -> np.ndarray:
        if hasattr(self.price_process, 'generate_ending_distribution'):
            s_maturity = self._calculate_process_ending_distribution(num_points=num_points)
        else:
            s_maturity = self._calculate_process_ending_simulation()
        return s_maturity

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
        {\\partial S^2}`). The gamma of the European option is calculated as:

        :math:`\\small \\displaystyle \\Gamma = \\frac{e^{-qt} N'(d_1)}{S \\sigma \\sqrt{t}}`

        Where :math:`N'(x)` represents the PDF of the standard Gaussian distribution and :math:`d_1` is defined in
        its own method.

        :return: Gamma calculated for the European option.
        """
        pass

    def get_vega(self, scale: float = 1.) -> float:
        """
        The vega of a European option represents the expected change in the price of the option as a response to
        changes in the volatility of the underlying (:math:`\\small \\displaystyle \\frac{\\partial V}
        {\\partial \\sigma}`). The vega of the European option is calculated as:

        :math:`\\small \\displaystyle Vega = S \\sqrt{t} e^{-qt} N'(d_1)`

        Where :math:`N'(x)` represents the PDF of the standard Gaussian distribution and :math:`d_1` is defined in
        its own method.

        As presented above, :math:`Vega` measures the decrease in value in the European option as volatility changes
        a full 100%. Usually this measure is too large to be useful in usual European options, where the underlying
        volatility varies within narrower ranges. For this reason, :math:`Vega` is usually divided by a factor of
        100 to represent changes in value caused by unit percent changes in volatility.

        :param scale: scale of the vega, 1 equals a 100% change in volatility, while 0.01 equals a 1% change.
        :return: Vega calculated for the European option.
        """
        pass


class EuropeanCall(EuropeanOption):
    """
    Implementation of the European Call option valuation by using the test_analytical expression. The European Call
    option provides the holder the right, but not the obligation, to buy an underlying asset, at a predetermined
    strike price on a specified expiration date.
    """

    def price(self, num_points: int = 500_000) -> float:
        """
        The price of the European Call option is calculated as:

        :math:`\\small \\displaystyle C = S e^{-q t}N(d_1) - K e^{-r t}N(d_2)`

        Where :math:`N(x)` represents the CDF of the standard Gaussian distribution and :math:`d_1` and :math:`d_2`
        are defined in their respective methods.

        :return: Price calculated for the European Call option.
        """
        s_maturity = self.get_underlying_at_maturity(num_points=num_points)
        price = np.exp(-self.r * self.t) * np.maximum(s_maturity - self.k, 0).mean()
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
        delta = norm.cdf(d1) * exp(-self.q * self.t)
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
        first_element = (self.sigma * self.s * exp(-self.q * self.t) * norm.pdf(d1)) / (2 * sqrt(self.t))
        second_element = self.q * self.s * exp(-self.q * self.t) * norm.cdf(d1)
        third_element = self.r * self.k * exp(-self.r * self.t) * norm.cdf(d2)
        theta = (- first_element + second_element - third_element) / periods
        return theta

    def get_rho(self, target: str = 'r', scale: float = 1.) -> float:
        """
        The rho of a European option represents the expected change in the price of the option as a response to
        changes in the risk-free interest rate (:math:`\\small \\displaystyle \\frac{\\partial V} {\\partial r}`).
        The rho of the European Call option is calculated as:

        :math:`\\small \\displaystyle \\rho = K t e^{-rt} N(d_2)`

        Where :math:`N(x)` and :math:`N'(x)` represents respectively the CDF and PDF of the standard Gaussian
        distribution and :math:`d_1` and :math:`d_2` are defined in their respective methods.

        Sometimes rho is also used to measure the expected change in the price of the option as a response to
        changes in the dividend yield of the underlying (:math:`\\small \\displaystyle \\frac{\\partial V}
        {\\partial q}`). In this case the formula is:

        :math:`\\small \\displaystyle \\rho = -t S e^{-qt} N(d_1)`

        As presented above, :math:`\\rho` measures the decrease in value in the European option as the interest rate
        or the dividend yield change a full 100%. Usually this measure is too large to be useful in usual European
        options, where these metrics vary within narrower ranges. For this reason, :math:`\\rho` is usually divided
        by a factor of 100 to represent changes in value caused by unit percent changes in the rates.

        :param target: string signalling whether to get rho relative to interest rate ('r') or dividend yield ('q').
        :param scale: scale of the rho, 1 equals a 100% change in the interest rate, while 0.01 equals a 1% change.
        :return: rho calculated for the European Call option.
        """
        if target not in ['r', 'q']:
            raise ValueError(f'Target <{target}> not valid, accepted values are r and q')
        if target == 'r':
            d2 = self.get_d2()
            rho = (self.k * self.t * exp(-self.r * self.t) * norm.cdf(d2)) * scale
        else:
            d1 = self.get_d2()
            rho = (- self.t * self.s * exp(-self.q * self.t) * norm.cdf(d1)) * scale
        return rho


class EuropeanPut(EuropeanOption):
    """
    Implementation of the European Put option valuation by using the test_analytical expression. The European Put option
    provides the holder the right, but not the obligation, to sell an underlying asset, at a predetermined strike price
    on a specified expiration date.
    """

    def price(self, num_points: int = 500_000) -> float:
        """
        The price of the European Put option is calculated as:

        :math:`\\small \\displaystyle P = K e^{-r t}N(-d_2) - S e^{-q t}N(-d_1)`

        Where :math:`N(x)` represents the CDF of the standard Gaussian distribution and :math:`d_1` and :math:`d_2`
        are defined in their respective methods.

        :return: Price calculated for the European Put option.
        """
        s_maturity = self.get_underlying_at_maturity(num_points=num_points)
        price = np.exp(-self.r * self.t) * np.maximum(self.k - s_maturity, 0).mean()
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
        delta = (norm.cdf(d1) - 1) * exp(-self.q * self.t)
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
        first_element = (self.s * self.sigma * exp(-self.q * self.t) * norm.pdf(d1)) / (2 * sqrt(self.t))
        second_element = self.r * self.k * exp(-self.r * self.t) * norm.cdf(-d2)
        third_element = self.q * self.s * exp(-self.q * self.t) * norm.cdf(-d1)
        theta = (- first_element + second_element - third_element) / periods
        return theta

    def get_rho(self, target: str = 'r', scale: float = 1.) -> float:
        """
        The rho of a European option represents the expected change in the price of the option as a response to
        changes in the risk-free interest rate (:math:`\\small \\displaystyle \\frac{\\partial V} {\\partial r}`).
        The rho of the European Call option is calculated as:

        :math:`\\small \\displaystyle \\rho = K t e^{-rt} N(d_2)`

        Where :math:`N(x)` and :math:`N'(x)` represents respectively the CDF and PDF of the standard Gaussian
        distribution and :math:`d_1` and :math:`d_2` are defined in their respective methods.

        Sometimes rho is also used to measure the expected change in the price of the option as a response to
        changes in the dividend yield of the underlying (:math:`\\small \\displaystyle \\frac{\\partial V}
        {\\partial q}`). In this case the formula is:

        :math:`\\small \\displaystyle \\rho = -t S e^{-qt} N(d_1)`

        As presented above, :math:`\\rho` measures the decrease in value in the European option as the interest rate
        or the dividend yield change a full 100%. Usually this measure is too large to be useful in usual European
        options, where these metrics vary within narrower ranges. For this reason, :math:`\\rho` is usually divided
        by a factor of 100 to represent changes in value caused by unit percent changes in the rates.

        :param target: string signalling whether to get rho relative to interest rate ('r') or dividend yield ('q').
        :param scale: scale of the rho, 1 equals a 100% change in the interest rate, while 0.01 equals a 1% change.
        :return: rho calculated for the European Call option.
        """
        if target not in ['r', 'q']:
            raise ValueError(f'Target <{target}> not valid, accepted values are r and q')
        if target == 'r':
            d2 = self.get_d2()
            rho = (- self.k * self.t * exp(-self.r * self.t) * norm.cdf(-d2)) * scale
        else:
            d1 = self.get_d2()
            rho = (self.t * self.s * exp(-self.q * self.t) * norm.cdf(-d1)) * scale
        return rho
