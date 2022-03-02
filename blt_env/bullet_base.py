from abc import ABCMeta, abstractmethod


class BulletEnv(metaclass=ABCMeta):

    def __init__(self, is_gui: bool = True):
        self._is_gui = is_gui

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError

    def get_is_gui(self) -> bool:
        return self._is_gui
