from abc import ABCMeta, abstractmethod

from blt_env.drone import DroneBltEnv


class DroneEnvControl(metaclass=ABCMeta):

    def __init__(self, env: DroneBltEnv):
        self._env = env
        self._control_counter = 0

    @abstractmethod
    def compute_control(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        self._control_counter = 0

    def get_control_counter(self) -> int:
        return self._control_counter
