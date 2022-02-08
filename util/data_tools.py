from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, astuple, asdict
import math
from enum import IntEnum, Enum

import numpy as np


def get_values_list(data: dataclass) -> Optional[List]:
    if data is None:
        return
    return list(astuple(data))


def get_values_csv(data: dataclass) -> Optional[str]:
    v_list = get_values_list(data)
    if v_list is None:
        return
    return ','.join(map(str, v_list))


def get_keys_list(data: dataclass) -> Optional[List]:
    if data is None:
        return
    data_dic = asdict(data)
    return list(data_dic.keys())


def get_keys_csv(data: dataclass) -> Optional[str]:
    k_list = get_keys_list(data)
    if k_list is None:
        return
    return ','.join(map(str, k_list))


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


@dataclass
class DroneKinematicInfo(object):
    pos: np.ndarray = np.zeros(3)  # position
    quat: np.ndarray = np.zeros(4)  # quaternion
    rpy: np.ndarray = np.zeros(3)  # roll, pitch and yaw
    vel: np.ndarray = np.zeros(3)  # linear velocity
    ang_vel: np.ndarray = np.zeros(3)  # angular velocity
    rpy_rates: np.ndarray = np.zeros(3)  # roll, pitch, and yaw rates


@dataclass
class DroneControlTarget(object):
    pos: np.ndarray = np.zeros(3)  # position
    vel: np.ndarray = np.zeros(3)  # linear velocity
    rpy: np.ndarray = np.zeros(3)  # roll, pitch and yaw
    ang_vel: np.ndarray = np.zeros(3)  # angular velocity


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
