
from abc import ABCMeta, abstractmethod


class CallableDeviceBase(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, **kwargs):  # data_load
        """Retrieve data from the input source and return an object."""
        raise NotImplementedError

    @abstractmethod
    def open(self):
        """Open the measurement dev."""
        pass

    @abstractmethod
    def close(self):
        """Close the measurement dev.
        """
        pass
