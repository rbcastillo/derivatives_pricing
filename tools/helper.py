from math import exp
from typing import Union


class Balance:
    """
    Auxiliary class to centralize the tasks of calculating future value and present value of a monetary balance
    under constant interest rate :math:`r` considering a capitalization/discount period of :math:`t`.
    """

    def __init__(self, amount: Union[int, float]) -> None:
        """
        Method to initialize the Balance class with its reference amount. For the capitalize method, this is the
        current amount at time :math:`t_0` that will be capitalized over time to obtain its value at time
        :math:`T=(t_0 + t)`. For the discount method, this will be the amount at time :math:`T=(t_0 + t)` whose
        present value at time :math:`t_0` will be calculated.

        :param amount: reference amount to be considered in the calculations.
        """
        self.amount = amount

    def capitalize(self, r: float, t: Union[int, float], compounding: str) -> float:
        """
        Method to capitalize the balance amount at interest rate :math:`r` during a period :math:`t`. Please, note
        that :math:`r` must be measured in the same time units as :math:`t`, i.e., if :math:`t` is measured in years,
        :math:`r` should be the annualized interest rate.

        The following capitalization methods are supported:

        [Discrete] :math:`\\small \\displaystyle Balance_T = Balance_{t_0} · (1 + r)^{t}`

        [Continuous] :math:`\\small \\displaystyle Balance_T = Balance_{t_0} · e^{rt}`

        :param r: rate of return considered in the balance capitalization.
        :param t: time during which the capitalization process is performed.
        :param compounding: type of compounding used, accepted values are discrete and continuous.
        :return: capitalized future value of the current balance amount.
        """
        if compounding == 'discrete':
            amount_fv = self.amount * (1 + r) ** t
        elif compounding == 'continuous':
            amount_fv = self.amount * exp(r * t)
        else:
            raise ValueError(f'Compounding type <{compounding}> not recognized, use discrete or continuous')
        return amount_fv

    def discount(self, r: float, t: Union[int, float], compounding: str) -> float:
        """
        Method to discount to present value a future balance amount at interest rate :math:`r` during a
        period :math:`t`. Please, note that :math:`r` must be measured in the same time units as :math:`t`,
        i.e., if :math:`t` is measured in years, :math:`r` should be the annualized interest rate.

        The following capitalization methods are supported:

        [Discrete] :math:`\\small \\displaystyle Balance_T = Balance_{t_0} · (1 + r)^{-t}`

        [Continuous] :math:`\\small \\displaystyle Balance_T = Balance_{t_0} · e^{-rt}`

        :param r: rate of return considered in the balance discount.
        :param t: time during which the discount process is performed.
        :param compounding: type of compounding used, accepted values are discrete and continuous.
        :return: discounted present value of the future balance amount.
        """
        if compounding == 'discrete':
            amount_pv = self.amount * (1 + r) ** -t
        elif compounding == 'continuous':
            amount_pv = self.amount * exp(-r * t)
        else:
            raise ValueError(f'Compounding type <{compounding}> not recognized, use discrete or continuous')
        return amount_pv
