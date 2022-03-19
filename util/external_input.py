import pybullet as p
from typing import List
from blt_env.drone import DroneBltEnv
from util.data_definition import DroneControlTarget

import numpy as np


def key_code_list2binary(code_list: List[int]):
    """
    Parameters
    ----------
    code_list: e.g. [100, 107, 65295]

    Returns
    -------
        e.g. 0b10000100(binary code)
    """
    """
    Parameters
    ----------
    code_list: e.g. [100, 107, 65295]
    """
    ''' key2binary map
    up          down        ccw         cw          left        right       back        forward
    d           c           z           x           h           k           j           u
    0b1000_0000 0b0100_0000 0b0010_0000 0b0001_0000 0b0000_1000 0b0000_0100 0b0000_0010 0b0000_0001
    128          64          32          16          8           4           2           1    
    '''
    key2binary_map = {
        'd': 0b1000_0000,  # up
        'c': 0b0100_0000,  # down
        'z': 0b0010_0000,  # ccw
        'x': 0b0001_0000,  # cw
        'h': 0b0000_1000,  # left
        'k': 0b0000_0100,  # right
        'j': 0b0000_0010,  # back
        'u': 0b0000_0001,  # forward
    }
    input_keys = [chr(k) for k in code_list]
    input_keys_bin = [key2binary_map.get(k, 0) for k in input_keys]
    return sum(input_keys_bin)


def get_key_pressed():
    pressed_keys = []
    events = p.getKeyboardEvents()
    key_codes = events.keys()
    for key in key_codes:
        pressed_keys.append(key)
    return pressed_keys


class KeyboardInputCtrl(object):
    """
    Compute the target of the drone control data (position, velocity and roll-pitch-yaw)
    in the next simulation step from the keyboard input information.

        https://github.com/toritamantaro/pybullet-drone-self-study

        Based on code written by taront.

    """

    def __init__(
            self,
            blt_env: DroneBltEnv,
            nth_drone: int = 0,
    ):
        self._blt_env = blt_env
        self._ctrl_drone_id = nth_drone
        self._ctrl_time_step = self._blt_env.get_sim_time_step() * self._blt_env.get_aggr_phy_steps()
        self._dp = self._blt_env.get_drone_properties()
        self._max_speed = self._dp.max_speed_kmh * 1000 / (60 * 60)
        self._max_step_move = self._ctrl_time_step * self._max_speed
        self._delta_xy = 1. * self._max_step_move
        self._delta_z = 0.5 * self._max_step_move
        self._delta_c = 6. * self._ctrl_time_step * np.pi
        self._increase_rate_with_vel = 2.5

        current_ki = self._blt_env.get_drones_kinematic_info()[self._ctrl_drone_id]
        self._ctrl_targets = DroneControlTarget(
            pos=current_ki.pos,
            vel=current_ki.vel,
            rpy=current_ki.rpy,
        )

    @staticmethod
    def is_moving(vel: np.ndarray) -> bool:
        threshold = 0.0001
        is_moving = True if np.linalg.norm(vel) > threshold else False
        return is_moving

    def get_ctrl_target(self):
        if self._blt_env.get_sim_counts() % self._blt_env.get_aggr_phy_steps() != 0:
            return self._ctrl_targets

        # get drone kinematic information
        kis = self._blt_env.get_drones_kinematic_info()
        ki = kis[self._ctrl_drone_id]

        # get key press information
        key_codes = get_key_pressed()
        key_info = key_code_list2binary(key_codes)

        # get control directions.
        d_z = float((key_info >> 7) & 0b1) - float((key_info >> 6) & 0b1)
        d_y = float((key_info >> 3) & 0b1) - float((key_info >> 2) & 0b1)
        d_x = -float((key_info >> 1) & 0b1) + float((key_info >> 0) & 0b1)
        d_c = float((key_info >> 5) & 0b1) - float((key_info >> 4) & 0b1)

        if key_info:
            if not np.all([d_x, d_y, d_z]):
                # compute target coordinate
                target_diff = np.array([
                    self._delta_xy * d_x,
                    self._delta_xy * d_y,
                    self._delta_z * d_z,
                ])

                # Constraints on xy diagonal movement
                if (xy_norm := np.linalg.norm(target_diff[:2])) != 0:
                    target_diff[:2] = 0.5 * (self._delta_xy * target_diff[:2] / xy_norm)

                kis = self._blt_env.get_drones_kinematic_info()
                ki = kis[self._ctrl_drone_id]

                rotation = np.array(p.getMatrixFromQuaternion(ki.quat)).reshape(3, 3)
                world_diff = np.dot(rotation, target_diff)

                cos_sim_denom = np.linalg.norm(world_diff) * np.linalg.norm(ki.vel)
                cos_sim = np.dot(world_diff, ki.vel) / cos_sim_denom if cos_sim_denom != 0 else -1
                if cos_sim > 0. and self.is_moving(ki.vel):
                    world_diff = world_diff * self._increase_rate_with_vel * (np.abs(ki.vel) + np.ones(3))

                t_pos = ki.pos + world_diff
                self._ctrl_targets.pos[0:2] = np.array(t_pos[0:2])

                if d_z != 0:
                    self._ctrl_targets.pos[2] = np.array(t_pos[2])

            if d_c != 0:
                # compute target yaw
                t_yaw = ki.rpy[2] + self._delta_c * d_c
                t_ryp = [0, 0, t_yaw]
                self._ctrl_targets.rpy = np.array(t_ryp)

        else:
            # Deceleration and stopping process
            if self.is_moving(ki.vel):
                target_speed_reduction_rate = 0.1
                self._ctrl_targets.vel = target_speed_reduction_rate * ki.vel
                self._ctrl_targets.pos = self._ctrl_targets.pos + ki.vel * self._blt_env.get_sim_time_step()
            else:
                self._ctrl_targets.vel = np.zeros(3)
                self._ctrl_targets.pos = ki.pos

            self._ctrl_targets.rpy = np.array([0, 0, ki.rpy[2]])

        return self._ctrl_targets
