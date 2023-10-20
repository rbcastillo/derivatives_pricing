from abc import ABC, abstractmethod
from math import sqrt, exp, log

import numpy as np
from scipy.optimize import fmin
from scipy.stats import norm
from typing import Union


class EuropeanOption(ABC):
    """
    Generic abstract implementation of the European option object populated with the key auxiliary methods
    shared between Call and Put options. The pricing method is defined as an abstract method to be implemented
    in the Call and Put specific objects depending on their features.
    """

    __slots__ = ['s', 'k', 'r', 't', 'sigma', 'q', 'd1', 'd2', '_ignore']

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
        super().__setattr__('s', s)
        super().__setattr__('k', k)
        super().__setattr__('r', r)
        super().__setattr__('t', t)
        super().__setattr__('sigma', sigma)
        super().__setattr__('q', q)
        super().__setattr__('d1', None)
        super().__setattr__('d2', None)
        super().__setattr__('_ignore', ['d1', 'd2', '_ignore'])

    def __str__(self) -> str:
        """
        Method to create a string representation of the object. This method outputs the object name and the
        relevant parameters with their values.

        :return: string representation of the object.
        """
        attr_values = {attr: getattr(self, attr) for attr in self.__slots__ if attr not in self._ignore}
        text_output = f'{self.__class__.__name__} object with parameters {attr_values}'
        return text_output

    def __setattr__(self, key, value) -> None:
        """
        Method to manage the process of setting and updating object parameters. Only attributes declared in the
        __init__ method that do not appear in the _ignore attribute are allowed to be updated. This avoids adding
        new attributes or modifying attributes that should only be modified inside the object as a result of
        certain operations.

        :param key: parameter to be added or updated.
        :param value: value to be assigned to the parameter.
        :return: None
        """
        if key not in self.__slots__ or key in self._ignore:
            valid_attrs = set(self.__slots__) - set(self._ignore)
            raise ValueError(f'Attribute name {key} is not recognized, use values in {valid_attrs}')
        super().__setattr__(key, value)
        if key not in self._ignore:
            super().__setattr__('d1', None)
            super().__setattr__('d2', None)

    def update_params(self, **kwargs) -> None:
        """
        This method allows updating some or all of the relevant input for the European option class. The kwargs used
        in the method must match those in the __init__ method.

        :param kwargs: keyword parameter(s) and value(s) to be updated.
        :return: None.
        """
        for param, value in kwargs.items():
            setattr(self, param, value)

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
            d1_numerator = log(self.s / self.k) + (self.r - self.q + (self.sigma ** 2) / 2) * self.t
            d1_denominator = self.sigma * sqrt(self.t)
            super().__setattr__('d1', d1_numerator / d1_denominator)
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
            super().__setattr__('d2', d1 - self.sigma * sqrt(self.t))
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
        {\\partial S^2}`). The gamma of the European option is calculated as:

        :math:`\\small \\displaystyle \\Gamma = \\frac{e^{-qt} N'(d_1)}{S \\sigma \\sqrt{t}}`

        Where :math:`N'(x)` represents the PDF of the standard Gaussian distribution and :math:`d_1` is defined in
        its own method.

        :return: Gamma calculated for the European option.
        """
        d1 = self.get_d1()
        gamma = exp(-self.q * self.t) * norm.pdf(d1) / (self.s * self.sigma * sqrt(self.t))
        return gamma

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
        d1 = self.get_d1()
        vega = (self.s * sqrt(self.t) * exp(-self.q * self.t) * norm.pdf(d1)) * scale
        return vega

    def _get_price_distance(self, sigma: float, target_price: Union[int, float]):
        """
        Auxiliary method for the calculation of the implied volatility. This method calculates the price of the
        option for a specified volatility of the underlying and returns the absolute difference with the target
        price for the option.

        :param sigma: volatility level for the underlying security.
        :param target_price: target price observed in the market.
        :return: absolute difference between the theoretical price for the given volatility and the target price.
        """
        self.sigma = sigma[0] if isinstance(sigma, np.ndarray) else sigma
        price = self.price()
        return abs(price - target_price)

    def calc_implied_vol(self, price: Union[int, float], disp: bool = False) -> float:
        """
        The implied volatility of a European option is the value for the volatility of the underlying asset that
        makes the theoretical valuation of the option equal to the observed market value.

        Since there is no closed-form expression for the implied volatility of the asset, it must be obtained from
        the analytics expression for the value of the European option using numerical methods. In this implementation
        the scipy version of the Nelder-Mead simplex algorithm is used.

        :param price: price for the European option observed in the market.
        :param disp: set to True to print convergence messages.
        :return: implicit volatility for the given market price of the European option.
        """
        base_sigma = self.sigma
        implicit_volatility = fmin(self._get_price_distance, x0=base_sigma, args=(price,), disp=disp)[0]
        self.__setattr__('sigma', base_sigma)
        return implicit_volatility


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
        price = self.s * exp(-self.q * self.t) * norm.cdf(d1) - self.k * exp(-self.r * self.t) * norm.cdf(d2)
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
            raise ValueError(f'Target {target} not valid, accepted values are r and q')
        if target == 'r':
            d2 = self.get_d2()
            rho = (self.k * self.t * exp(-self.r * self.t) * norm.cdf(d2)) * scale
        else:
            d1 = self.get_d2()
            rho = (- self.t * self.s * exp(-self.q * self.t) * norm.cdf(d1)) * scale
        return rho


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
        price = self.k * exp(-self.r * self.t) * norm.cdf(-d2) - self.s * exp(-self.q * self.t) * norm.cdf(-d1)
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
            raise ValueError(f'Target {target} not valid, accepted values are r and q')
        if target == 'r':
            d2 = self.get_d2()
            rho = (- self.k * self.t * exp(-self.r * self.t) * norm.cdf(-d2)) * scale
        else:
            d1 = self.get_d2()
            rho = (self.t * self.s * exp(-self.q * self.t) * norm.cdf(-d1)) * scale
        return rho


if __name__ == '__main__':
    call = EuropeanCall(s=100, k=120, r=0.05, t=1, sigma=0.2, q=0.02)
    print(call)
    print('Call:', round(call.price(), 9))
    print('Implied vol:', round(call.calc_implied_vol(4.3749), 9))
    print('Call delta:', round(call.get_delta(), 9))
    print('Call gamma:', round(call.get_gamma(), 9))
    print('Call theta:', round(call.get_theta(1), 9))
    print('Call vega:', round(call.get_vega(), 9))
    print('Call rho:', round(call.get_rho(), 9))

    print('\n')
    put = EuropeanPut(s=100, k=120, r=0.05, t=1, sigma=0.2, q=0.02)
    print(put)
    print('Put:', round(put.price(), 9))
    print('Implied vol:', round(put.calc_implied_vol(20.5026), 9))
    print('Put delta:', round(put.get_delta(), 9))
    print('Put gamma:', round(put.get_gamma(), 9))
    print('Put theta:', round(put.get_theta(1), 9))
    print('Put vega:', round(put.get_vega(), 9))
    print('Put rho:', round(put.get_rho(), 9))
