from typing import List, Optional
from dataclasses import dataclass, field
from enum import IntEnum, Enum

import numpy as np

"""
# @dataclass(init=False, frozen=True)
@dataclass(frozen=True)
class FooBarConstants(object):
    bar = 'b_val'
    foo = 'f_val'

    # def __post_init__(self):
    #     self.bar = 'bar_value'  # FrozenInstanceError
    #     object.__setattr__(self, 'foo', 'foo_value') # not get error

"""


@dataclass(init=False, frozen=True)
class SharedConstants(object):
    AGGR_PHY_STEPS: int = 5
    # for default setting
    SUCCESS_MODEL_FILE_NAME: str = 'success_model.zip'
    DEFAULT_OUTPUT_DIR_PATH: str = './result'
    DEFAULT_DRONE_FILE_PATH: str = './assets/drone_x_01.urdf'
    DEFAULT_DRONE_TYPE_NAME: str = 'x'


class DroneType(IntEnum):
    OTHER = 0
    QUAD_PLUS = 1
    QUAD_X = 2


class PhysicsType(Enum):
    """Physics implementations enumeration class."""
    PYB = "pyb"  # Base PyBullet physics update
    DYN = "dyn"  # Update with an explicit model of the dynamics
    PYB_GND = "pyb_gnd"  # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"  # PyBullet physics update with drag
    PYB_DW = "pyb_dw"  # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw"  # PyBullet physics update with ground effect, drag, and downwash


class ActionType(Enum):
    """Action type enumeration class."""
    RPM = "rpm"  # RPMS
    FORCE = "for"  # Desired thrust and torques (force)
    PID = "pid"  # PID control
    VEL = "vel"  # Velocity input (using PID control)
    TUN = "tun"  # Tune the coefficients of a PID controller
    ONE_D_RPM = "one_d_rpm"  # 1D (identical input to all motors) with RPMs
    ONE_D_FORCE = "one_d_for"  # 1D (identical input to all motors) with desired thrust and torques
    ONE_D_PID = "one_d_pid"  # 1D (identical input to all motors) with PID control


class RlAlgorithmType(Enum):
    """Reinforcement Learning type enumeration class."""
    A2C = 'a2c'
    PPO = 'ppo'
    SAC = 'sac'
    TD3 = 'td3'
    DDPG = 'ddpg'


class ObservationType(Enum):
    """Observation type enumeration class."""
    KIN = "kin"  # Kinematics information (pose, linear and angular velocities)
    RGB = "rgb"  # RGB camera capture in each drone's POV


@dataclass(frozen=True)
class DroneForcePIDCoefficients(object):
    P_for: np.ndarray = None  # force
    I_for: np.ndarray = None
    D_for: np.ndarray = None
    P_tor: np.ndarray = None  # torque
    I_tor: np.ndarray = None
    D_tor: np.ndarray = None


@dataclass
class DroneKinematicsInfo(object):
    pos: np.ndarray = np.zeros(3)  # position
    quat: np.ndarray = np.zeros(4)  # quaternion
    rpy: np.ndarray = np.zeros(3)  # roll, pitch and yaw
    vel: np.ndarray = np.zeros(3)  # linear velocity
    ang_vel: np.ndarray = np.zeros(3)  # angular velocity


@dataclass
class DroneControlTarget(object):
    pos: np.ndarray = np.zeros(3)  # position
    vel: np.ndarray = np.zeros(3)  # linear velocity
    rpy: np.ndarray = np.zeros(3)  # roll, pitch and yaw
    rpy_rates: np.ndarray = np.zeros(3)  # roll, pitch, and yaw rates


@dataclass
class DroneProperties(object):
    """
    The drone parameters.

    kf : It is the proportionality constant for thrust, and thrust is proportional to the square of rotation speed.
    km : It is the proportionality constant for torque, and torque is proportional to the square of rotation speed.

    """
    type: int = 1  # The drone type 0:OTHER 1:QUAD_PLUS 2:QUAD_X
    g: float = 9.8  # gravity acceleration
    m: Optional[float] = None  # Mass of the drone.
    l: Optional[float] = None  # Length of the arm of the drone's rotor mount.
    thrust2weight_ratio: Optional[float] = None
    ixx: float = 0
    iyy: float = 0
    izz: float = 0
    J: np.ndarray = np.array([])
    J_inv: np.ndarray = np.array([])
    kf: Optional[float] = None  # The proportionality constant for thrust.
    km: Optional[float] = None  # The proportionality constant for torque.
    collision_h: Optional[float] = None
    collision_r: Optional[float] = None
    collision_shape_offsets: List[float] = field(default_factory=list)
    collision_z_offset: float = None
    max_speed_kmh: Optional[float] = None
    gnd_eff_coeff: Optional[float] = None
    prop_radius: Optional[float] = None
    drag_coeff_xy: float = 0
    drag_coeff_z: float = 0
    drag_coeff: np.ndarray = None
    dw_coeff_1: Optional[float] = None
    dw_coeff_2: Optional[float] = None
    dw_coeff_3: Optional[float] = None
    # compute after determining the drone type
    gf: float = 0  # gravity force
    hover_rpm: float = 0
    max_rpm: float = 0
    max_thrust: float = 0
    max_xy_torque = 0
    max_z_torque = 0
    grand_eff_h_clip = 0  # The threshold height for ground effects.
    A: np.ndarray = np.array([])
    inv_A: np.ndarray = np.array([])
    B_coeff: np.ndarray = np.array([])
    Mixer: np.ndarray = np.ndarray([])  # use for PID control

    def __post_init__(self):
        self.J = np.diag([self.ixx, self.iyy, self.izz])
        self.J_inv = np.linalg.inv(self.J)
        self.collision_z_offset = self.collision_shape_offsets[2]
        self.drag_coeff = np.array([self.drag_coeff_xy, self.drag_coeff_xy, self.drag_coeff_z])
        self.gf = self.g * self.m
        self.hover_rpm = np.sqrt(self.gf / (4 * self.kf))
        self.max_rpm = np.sqrt((self.thrust2weight_ratio * self.gf) / (4 * self.kf))
        self.max_thrust = (4 * self.kf * self.max_rpm ** 2)
        if self.type == 2:  # QUAD_X
            self.max_xy_torque = (2 * self.l * self.kf * self.max_rpm ** 2) / np.sqrt(2)
            self.A = np.array([[1, 1, 1, 1], [1 / np.sqrt(2), 1 / np.sqrt(2), -1 / np.sqrt(2), -1 / np.sqrt(2)],
                               [-1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2), -1 / np.sqrt(2)], [-1, 1, -1, 1]])
            self.Mixer = np.array([[.5, -.5, -1], [.5, .5, 1], [-.5, .5, -1], [-.5, -.5, 1]])
        elif self.type in [0, 1]:  # QUAD_PLUS, OTHER
            self.max_xy_torque = (self.l * self.kf * self.max_rpm ** 2)
            self.A = np.array([[1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1]])
            self.Mixer = np.array([[0, -1, -1], [+1, 0, 1], [0, 1, -1], [-1, 0, 1]])
        self.max_z_torque = 2 * self.km * self.max_rpm ** 2
        self.grand_eff_h_clip = 0.25 * self.prop_radius * np.sqrt(
            (15 * self.max_rpm ** 2 * self.kf * self.gnd_eff_coeff) / self.max_thrust)
        self.inv_A = np.linalg.inv(self.A)
        self.B_coeff = np.array([1 / self.kf, 1 / (self.kf * self.l), 1 / (self.kf * self.l), 1 / self.km])
