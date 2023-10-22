from abc import ABC, abstractmethod


class FinancialProduct(ABC):
    """
    Base object to implement the main methods to be shared across all financial products implementations.
    """

    __slots__ = ['_ignore']

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
            valid_attrs = [attr for attr in self.__slots__ if attr not in self._ignore]
            raise ValueError(f'Attribute name <{key}> is not recognized, use values in {valid_attrs}')
        object.__setattr__(self, key, value)
        if key not in self._ignore:
            object.__setattr__(self, 'd1', None)
            object.__setattr__(self, 'd2', None)

    def update_params(self, **kwargs) -> None:
        """
        This method allows updating some or all of the relevant input for the European option class. The kwargs used
        in the method must match those in the __init__ method.

        :param kwargs: keyword parameter(s) and value(s) to be updated.
        :return: None.
        """
        for param, value in kwargs.items():
            setattr(self, param, value)