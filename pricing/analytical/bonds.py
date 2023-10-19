from abc import ABC, abstractmethod
from math import exp
from typing import Union


class Bond(ABC):

    def __init__(self, p: Union[int, float], r: float, t: Union[int, float], compounding: str) -> None:
        """

        :param p:
        :param r:
        :param t:
        :param compounding: type of compounding used, acceptable values are discrete and continuous.
        """
        if compounding not in ['discrete', 'continuous']:
            raise ValueError(f'Compounding type {compounding} not recognized, use discrete or continuous')
        self.p = p
        self.r = r
        self.t = t
        self.compounding = compounding

    def get_principal_repayment_present_value(self):
        if self.compounding == 'discrete':
            pv_p = self.p * (1 + self.r) ** -self.t
        elif self.compounding == 'discrete':
            pv_p = self.p * exp(-self.r * self.t)
        else:
            raise ValueError(f'Compounding type {self.compounding} not recognized, use discrete or continuous')
        return pv_p

    @abstractmethod
    def price(self):
        pass


class ZeroCouponBond(Bond):

    def price(self):
        price = self.get_principal_repayment_present_value()
        return price
