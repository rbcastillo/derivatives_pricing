import warnings
import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, Union, Collection, Any, Optional


class FinancialProduct(ABC):
    """
    Base object to implement the main methods to be shared across all financial products implementations.
    """

    __slots__ = []

    @abstractmethod
    def __init__(self) -> None:
        """
        Abstract method for the initialization of each specific object based on its implementation.
        """
        pass

    def __str__(self) -> str:
        """
        Method to create a string representation of the object. This method outputs the object name and the
        relevant parameters with their values.

        :return: string representation of the object.
        """
        attr_values = {attr: getattr(self, attr) for attr in self.__slots__ if not attr.startswith('_')}
        text_output = f'{self.__class__.__name__} object with parameters {attr_values}'
        return text_output

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
        if key not in self.__slots__ or key.startswith('_'):
            valid_attrs = [attr for attr in self.__slots__ if not attr.startswith('_')]
            if key not in self.__slots__:
                raise ValueError(f'Attribute name <{key}> is not recognized, use values in {valid_attrs}')
            if key.startswith('_'):
                raise ValueError(f'Attribute <{key}> is protected or private, use values in {valid_attrs}')
        else:
            object.__setattr__(self, key, value)

    def update_params(self, **kwargs: Any) -> None:
        """
        This method allows updating some or all of the relevant input for the object. The kwargs used in the
        method must match those in the __init__ method.

        :param kwargs: keyword parameter(s) and value(s) to be updated.
        :return: None.
        """
        for param, value in kwargs.items():
            setattr(self, param, value)


class StatisticalProcess(ABC):
    """
    Base object to implement the main methods to be shared across all statistical processes implementations.
    """

    __slots__ = ['size', 'periods', 'asset_attributes']

    @abstractmethod
    def __init__(self, size: Tuple[int, ...], periods: int) -> None:
        """
        Abstract method for the initialization of each specific object based on its implementation. The size
        attribute, needed across all statistical processes is loaded in this base method.

        The asset_attributes object is automatically created to store the name of the attributes that are
        associated to the asset dimension. This way, the appropriate review process is performed when these
        attributes are modified.

        :param size: tuple containing the dimensions of the output statistical process. The first value represents
            the number of time steps simulated, the second number (if present) represents the number of independent
            paths simulated and the third number (if present) represents the number of different assets simulated.
        :param periods: number of periods per reference time unit. The default is 252 trading days when one year is
            the reference time unit in which metrics such as returns and volatility are expressed.
        """
        object.__setattr__(self, 'size', self._manage_size(size))
        object.__setattr__(self, 'periods', periods)
        object.__setattr__(self, 'asset_attributes', [])

    @staticmethod
    def _manage_size(size: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Method to ensure that the size parameter is compliant with the logic of the process. The size parameter should
        meet the following criteria:

        - Length: the length of the parameter should be between 1 and 3 inclusive for it to have financial sense. The
          first dimension represents the number of time steps, the second the number of different paths and the third
          the number of asset simulations.
        - Zeros: the size tuple should not have zeros in the intermediate places, zeros at the end are removed reducing
          the length of the parameter.

        :param size: tuple containing the dimensions of the output statistical process.
        :return: adjusted tuple if needed, otherwise the original tuple is returned.
        """
        StatisticalProcess._manage_size_length(size)
        size = StatisticalProcess._manage_size_zeros(size)
        return size

    @staticmethod
    def _manage_size_length(size: Tuple[int, ...]) -> None:
        """
        Auxiliary method to ensure that the length of the size parameter is compliant with the logic of the process.
        The length of the parameter should be between 1 and 3 inclusive for it to have financial sense. The first
        dimension represents the number of time steps, the second the number of different paths and the third
        the number of asset simulations.

        :param size: tuple containing the dimensions of the output statistical process.
        :return: None.
        """
        if len(size) == 0:
            raise ValueError(f'Size <{size}> not valid, at least should have one dimension')
        elif len(size) > 3:
            raise ValueError(f'Size <{size}> not valid, maximum number of dimensions is 3')

    @staticmethod
    def _manage_size_zeros(size: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Method to ensure that the content of the size parameter is compliant with the logic of the process. The size
        tuple should not have zeros in the intermediate places, zeros at the end are removed reducing the length of
        the parameter.

        :param size: tuple containing the dimensions of the output statistical process.
        :return: adjusted tuple if needed, otherwise the original tuple is returned.
        """
        if 0 in size:
            if size[-1] == 0:
                size = StatisticalProcess._manage_size(size[:-1])
            else:
                raise ValueError(f'Size <{size}> not valid, zero length inner dimensions are not allowed')
        return size

    def _assign_parameters_with_asset_dimension(self, **kwargs: Union[int, float, Collection]) -> None:
        """
        Method to assign the parameters that may vary by simulated asset after checking that they are compliant
        with the expected formats. If a parameter needs some adjustment, it is also managed inside this method.

        :param kwargs: keyword parameter(s) and value(s) to be verified and assigned.
        :return: None.
        """
        for name, value in kwargs.items():
            StatisticalProcess._check_parameter_with_asset_dimension(name, value, self.size)
            value = StatisticalProcess._adjust_parameter_with_asset_dimension(value, self.size)
            self.asset_attributes.append(name)
            object.__setattr__(self, name, value)

    @staticmethod
    def _check_parameter_with_asset_dimension(name: str, value: Union[int, float, Collection[Union[int, float]]],
                                              size: Tuple[int, ...]) -> None:
        """
        Auxiliary method to check, in case that an input applicable across assets is a collection, whether it meets
        the size requirements to be consistent with the target statistical process output size. If the parameter
        meets the requirements, the method does nothing. Otherwise, a ValueError exception is raised.

        :param name: name of the parameter to analyze.
        :param value: value of the parameter to analyze.
        :param size: target statistical process output size.
        :return: None.
        """
        if hasattr(value, '__len__') and not isinstance(value, str):
            if not len(size) == 3:
                raise ValueError(f'<{name}> is a collection but there is no asset dimension in size: <{size}>')
            elif len(value) != size[2]:
                raise ValueError(f'Length of <{name}> <{value}> does not match the asset dimension in size: <{size}>')

    def __str__(self) -> str:
        """
        Method to create a string representation of the object. This method outputs the object name and the
        relevant parameters with their values.

        :return: string representation of the object.
        """
        attr_values = {attr: getattr(self, attr) for attr in self.__slots__ if not attr.startswith('_')}
        text_output = f'{self.__class__.__name__} object with parameters {attr_values}\n{self.__doc__}'
        return text_output

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
        if key not in self.__slots__ or key.startswith('_'):
            valid_attrs = [attr for attr in self.__slots__ if not attr.startswith('_')]
            if key not in self.__slots__:
                raise ValueError(f'Attribute name <{key}> is not recognized, use values in {valid_attrs}')
            if key.startswith('_'):
                raise ValueError(f'Attribute <{key}> is protected or private, use values in {valid_attrs}')
        else:
            value = self._adjust_special_attributes_before_setting(key, value)
            object.__setattr__(self, key, value)

    def update_params(self, **kwargs) -> None:
        """
        This method allows updating some or all of the relevant input for the European option class. The kwargs used
        in the method must match those in the __init__ method.

        :param kwargs: keyword parameter(s) and value(s) to be updated.
        :return: None.
        """
        for name, value in kwargs.items():
            setattr(self, name, value)

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Method to calculate the output for the statistical process, the implementation needs to be adjusted for each
        specific process in its class.

        The output dimensions follow this schema:

        - First axis (necessary): number of time steps simulated.
        - Second axis (if present): number of independent paths simulated for one asset.
        - Third axis (if present): number of different assets simulated.

        :return: generated output object for the statistical process.
        """
        pass
