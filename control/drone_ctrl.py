from scipy.optimize import nnls

import math
from typing import Tuple
from scipy.spatial.transform import Rotation
import numpy as np
import pybullet as p

from blt_env.drone import DroneBltEnv
from control.ctrl_base import DroneEnvControl
from util.data_definition import DroneForcePIDCoefficients
from util.data_definition import DroneKinematicsInfo, DroneControlTarget
from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


# logger.setLevel(DEBUG)  # for standalone debugging
# logger.addHandler(StreamHandler())  # for standalone debugging

def compute_rpm_by_nnls(
        thrust: float,
        x_torque: float,
        y_torque: float,
        z_torque: float,
        b_coeff: np.ndarray,
        a: np.ndarray,
        inv_a: np.ndarray = None,
):
    """
    Parameters
    ----------
    thrust : Desired thrust along the drone's z-axis
    x_torque : Desired drone's x-axis torque.
    y_torque : Desired drone's y-axis torque.
    z_torque : Desired drone's z-axis torque.
    b_coeff : (4,1)-shaped array of floats containing the coefficients to re-scale thrust and torques.
    a : (4, 4)-shaped array of floats containing the motors configuration.
    inv_a : (4, 4)-shaped array of floats, inverse of 'a'.

    Returns
    -------
        (4,)-shaped array of ints containing the desired squared RPMs of each rotor.
    """
    B = np.multiply(np.array([thrust, x_torque, y_torque, z_torque]), b_coeff)
    inv_a = np.linalg.inv(a) if inv_a is None else inv_a
    sq_rpm = np.dot(inv_a, B)
    # sq_rpmは4つのモータのそれぞれの回転数を2乗したものが格納されているという前提
    # ただし、普通に行列式を解くとマイナスになる解が出る場合がある。回転数を2乗したものを解としたいのでこれは都合が悪い。
    # そのような場合に、解の全てを正にするという制約を与えて回帰する方法としてNNLSがある。
    sq_rpm_nnls, res = None, None
    # NNLS if any of the desired ang vel is negative
    if np.min(sq_rpm) < 0:
        sq_rpm_nnls, res = nnls(a, B, maxiter=3 * a.shape[1])

    return sq_rpm, sq_rpm_nnls, res


class ForceControl(DroneEnvControl):
    """
    This is a reference from the following ...

        https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/utils/utils.py

    """

    def __init__(self, env: DroneBltEnv):
        super().__init__(env)
        self._is_gui = self._env.get_is_gui()
        self._time_step = self._env.get_sim_time_step()
        self._dp = self._env.get_drone_properties()
        self._g = self._dp.g
        self._mass = self._dp.m
        self._kf = self._dp.kf
        self._km = self._dp.km
        self._max_thrust = self._dp.max_thrust
        self._max_xy_torque = self._dp.max_xy_torque
        self._max_z_torque = self._dp.max_z_torque
        self._A = self._dp.A
        self._inv_A = self._dp.inv_A
        self._B_coeff = self._dp.B_coeff

    def value_check(
            self,
            thrust: float,
            x_torque: float,
            y_torque: float,
            z_torque: float,
            counter: int = 0,  # Simulation or control iteration, only used for printouts.
    ) -> None:
        if not self._is_gui:
            return

        name = self.__class__.__name__
        if thrust < 0 or thrust > self._max_thrust:
            mes = f"iter {counter} : in {name}, unfeasible thrust {thrust:.3f} \
            outside range [0, {self._max_thrust:.2f}]"
            logger.warning(mes)
        if np.abs(x_torque) > self._max_xy_torque:
            mes = f"iter {counter} : in {name}, unfeasible x_torque {x_torque:.2f} \
            outside range [{-self._max_xy_torque:.2f}, {self._max_xy_torque:.2f}]"
            logger.warning(mes)
        if np.abs(y_torque) > self._max_xy_torque:
            mes = f"iter {counter} : in {name}, unfeasible y_torque {y_torque:.2f} \
            outside range [{-self._max_xy_torque:.2f}, {self._max_xy_torque:.2f}]"
            logger.warning(mes)
        if np.abs(z_torque) > self._max_z_torque:
            mes = f"iter {counter} : in {name}, unfeasible z_torque {z_torque:.2f} \
            outside range [{-self._max_z_torque:.2f}, {self._max_z_torque:.2f}]"
            logger.warning(mes)

    def compute_control(
            self,
            target_thrust: float,
            target_x_torque: float,
            target_y_torque: float,
            target_z_torque: float,
    ):
        counts = self._env.get_sim_counts()
        self.value_check(target_thrust, target_x_torque, target_y_torque, target_z_torque, counts)

        sq_rpm, sq_nnls, res = compute_rpm_by_nnls(
            thrust=target_thrust,
            x_torque=target_x_torque,
            y_torque=target_y_torque,
            z_torque=target_z_torque,
            b_coeff=self._B_coeff,
            a=self._A,
            inv_a=self._inv_A,
        )

        if sq_nnls is None:
            return np.sqrt(sq_rpm), None, None

        if self._is_gui:
            name = self.__class__.__name__
            norm_1 = np.linalg.norm(sq_rpm)
            norm_2 = np.linalg.norm(sq_nnls)
            mes = f"""
            iter {counts} : in {name}, unfeasible squared rotor speeds, using NNLS.
            <Negative rotor speeds>
            sq. rotor speeds : [ {sq_rpm[0]}, {sq_rpm[1]}, {sq_rpm[2]}, {sq_rpm[3]} ]
            Normalized : [{sq_rpm[0] / norm_1}, {sq_rpm[1] / norm_1}, {sq_rpm[2] / norm_1}, {sq_rpm[3] / norm_1} ]
            <NNLS rotor speeds>
            sq. rotor speeds : [ {sq_nnls[0]}, {sq_rpm[1]}, {sq_rpm[2]}, {sq_rpm[3]} ] 
            Normalized : [ {sq_nnls[0] / norm_2}, {sq_rpm[1] / norm_2}, {sq_rpm[2] / norm_2}, {sq_rpm[3] / norm_2} ]  
            Residual : {res}
            """

        return np.sqrt(sq_nnls), None, None


class DSLPIDControl(DroneEnvControl):
    """
    This is a reference from the following ...

        https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/control/DSLPIDControl.py

    """

    def __init__(
            self,
            env: DroneBltEnv,
            pid_coeff: DroneForcePIDCoefficients,
    ):
        super().__init__(env)

        # PID constant parameters
        self._PID = pid_coeff

        self._pwm2rpm_scale = 0.2685
        self._pwm2rpm_const = 4070.3
        self._min_pwm = 20000
        self._max_pwm = 65535

        self._time_step = self._env.get_sim_time_step()
        self._dp = self._env.get_drone_properties()
        self._g = self._dp.g
        self._mass = self._dp.m
        self._kf = self._dp.kf
        self._km = self._dp.km
        self._Mixer = self._dp.Mixer
        self._gf = self._dp.gf

        # Initialized PID control variables
        self._last_rpy = np.zeros(3)
        self._last_pos_e = np.zeros(3)
        self._integral_pos_e = np.zeros(3)
        self._last_rpy_e = np.zeros(3)
        self._integral_rpy_e = np.zeros(3)

    def get_PID(self) -> DroneForcePIDCoefficients:
        return self._PID

    def set_PID(self, pid_coeff: DroneForcePIDCoefficients):
        self._PID = pid_coeff

    def reset(self):
        self._last_rpy = np.zeros(3)
        self._last_pos_e = np.zeros(3)
        self._integral_pos_e = np.zeros(3)
        self._last_rpy_e = np.zeros(3)
        self._integral_rpy_e = np.zeros(3)

    def compute_control_from_observation(
            self,
            control_timestep: float,
            obs_state: np.ndarray,
            target_position: np.ndarray,
            target_velocity: np.ndarray = np.zeros(3),
            target_rpy: np.ndarray = np.zeros(3),
            target_rpy_rates: np.ndarray = np.zeros(3),
    ) -> Tuple:
        """ Computes the PID control action (as RPMs) for a single drone.
        Parameters
        ----------
        control_timestep: The time step at which control is computed.
        obs_state: (20,)-shaped array of floats containing the current state of the drone.
        target_position: (3,1)-shaped array of floats containing the desired position.
        < The following are optionals. >
        target_velocity: (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_rpy: (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates: (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.
        """
        return self.compute_control(
            control_timestep=control_timestep,
            current_position=obs_state[0:3],
            current_quaternion=obs_state[3:7],
            current_velocity=obs_state[10:13],
            current_ang_velocity=obs_state[13:16],
            target_position=target_position,
            target_velocity=target_velocity,
            target_rpy=target_rpy,
            target_rpy_rates=target_rpy_rates,
        )

    def compute_control_from_kinematics(
            self,
            control_timestep: float,
            kin_state: DroneKinematicsInfo,
            ctrl_target: DroneControlTarget,
    ) -> Tuple:
        """ Computes the PID control action (as RPMs) for a single drone.
        Parameters
        ----------
        control_timestep: The time step at which control is computed.
        kin_state
        ctrl_target
        """
        return self.compute_control(
            control_timestep=control_timestep,
            current_position=kin_state.pos,
            current_quaternion=kin_state.quat,
            current_velocity=kin_state.vel,
            current_ang_velocity=kin_state.ang_vel,
            target_position=ctrl_target.pos,
            target_velocity=ctrl_target.vel,
            target_rpy=ctrl_target.rpy,
            target_rpy_rates=ctrl_target.rpy_rates,
        )

    def compute_control(
            self,
            control_timestep: float,
            current_position: np.ndarray,
            current_quaternion: np.ndarray,
            current_velocity: np.ndarray,
            current_ang_velocity: np.ndarray,
            target_position: np.ndarray,
            target_velocity: np.ndarray = np.zeros(3),
            target_rpy: np.ndarray = np.zeros(3),
            target_rpy_rates: np.ndarray = np.zeros(3),
    ) -> Tuple:
        """ Computes the PID control action (as RPMs) for a single drone.

        Parameters
        ----------
        control_timestep: The time step at which control is computed.
        current_position: (3,1)-shaped array of floats containing the current position.
        current_quaternion: (4,1)-shaped array of floats containing the current orientation as a quaternion.
        current_velocity: (3,1)-shaped array of floats containing the current velocity.
        current_ang_velocity: (3,1)-shaped array of floats containing the current angular velocity.
        target_position: (3,1)-shaped array of floats containing the desired position.
        < The following are optionals. >
        target_velocity: (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_rpy: (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates: (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.
        """

        thrust, computed_target_rpy, pos_e = self.dsl_pid_position_control(
            control_timestep,
            current_position,
            current_quaternion,
            current_velocity,
            target_position,
            target_velocity,
            target_rpy,
        )
        rpm = self.dsl_pid_attitude_control(
            control_timestep,
            thrust,
            current_quaternion,
            computed_target_rpy,
            target_rpy_rates,
        )
        cur_rpy = p.getEulerFromQuaternion(current_quaternion)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    def dsl_pid_position_control(
            self,
            control_timestep: float,
            current_position: np.ndarray,
            current_quaternion: np.ndarray,
            current_velocity: np.ndarray,
            target_position: np.ndarray,
            target_velocity: np.ndarray,
            target_rpy: np.ndarray,
    ) -> Tuple:
        cur_rotation = np.array(p.getMatrixFromQuaternion(current_quaternion)).reshape(3, 3)
        pos_e = target_position - current_position
        vel_e = target_velocity - current_velocity

        self._integral_pos_e = self._integral_pos_e + pos_e * control_timestep
        self._integral_pos_e = np.clip(self._integral_pos_e, -2., 2.)
        self._integral_pos_e[2] = np.clip(self._integral_pos_e[2], -0.15, 0.15)

        # PID target thrust
        target_thrust = np.multiply(self._PID.P_for, pos_e) \
                        + np.multiply(self._PID.I_for, self._integral_pos_e) \
                        + np.multiply(self._PID.D_for, vel_e) \
                        + np.array([0, 0, self._gf])
        scalar_thrust = max(0, np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self._kf)) - self._pwm2rpm_const) / self._pwm2rpm_scale
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()

        # Target rotation
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            logger.error(f"ctrl it {self._env.get_sim_counts()} in {self.__class__.__name__}, range [-pi, pi]")

        return thrust, target_euler, pos_e

    def dsl_pid_attitude_control(
            self,
            control_timestep: float,
            thrust: float,
            current_quaternion: np.ndarray,
            target_euler: np.ndarray,
            target_rpy_rates: np.ndarray,
    ) -> np.ndarray:
        cur_rotation = np.array(p.getMatrixFromQuaternion(current_quaternion)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(current_quaternion))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(),
                                                                                    target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        rpy_rates_e = target_rpy_rates - (cur_rpy - self._last_rpy) / control_timestep
        self._last_rpy = cur_rpy
        self._integral_rpy_e = self._integral_rpy_e - rot_e * control_timestep
        self._integral_rpy_e = np.clip(self._integral_rpy_e, -1500., 1500.)
        self._integral_rpy_e[0:2] = np.clip(self._integral_rpy_e[0:2], -1., 1.)
        # PID target torques
        target_torques = - np.multiply(self._PID.P_tor, rot_e) \
                         + np.multiply(self._PID.D_tor, rpy_rates_e) \
                         + np.multiply(self._PID.I_tor, self._integral_rpy_e)

        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self._Mixer, target_torques)
        pwm = np.clip(pwm, self._min_pwm, self._max_pwm)
        return self._pwm2rpm_scale * pwm + self._pwm2rpm_const

    def one_2_3_dim_interface(self, thrust):
        """
        :param thrust:
            Array of floats of length 1, 2, or 4 containing a desired thrust input.
        :return:
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.
        """
        dim = len(np.array(thrust))
        pwm = np.clip(
            (np.sqrt(np.array(thrust) / (self._kf * (4 / dim))) - self._pwm2rpm_const) / self._pwm2rpm_scale,
            self._min_pwm,
            self._max_pwm,
        )
        assert dim in [1, 2, 4], f'in one_2_3_dim_interface()'

        if dim in [1, 4]:
            return np.repeat(pwm, 4 / dim)
        elif dim == 2:
            return np.hstack([pwm, np.flip(pwm)])


if __name__ == "__main__":
    '''
    If you want to run this module by itself, try the following.

       $ python -m control.drone_ctrl

    '''

    import time
    from util.data_definition import DroneType, PhysicsType
    from util.data_logger import DroneDataLogger

    urdf_file = './assets/drone_p_01.urdf'
    drone_type = DroneType.QUAD_PLUS
    phy_mode = PhysicsType.PYB

    env = DroneBltEnv(
        urdf_path=urdf_file,
        d_type=drone_type,
        is_gui=True,
        phy_mode=phy_mode,
    )

    # controller
    pid = DroneForcePIDCoefficients(
        P_for=np.array([.4, .4, 1.25]),
        I_for=np.array([.05, .05, .05]),
        D_for=np.array([.2, .2, .5]),
        P_tor=np.array([70000., 70000., 60000.]),
        I_tor=np.array([.0, .0, 500.]),
        D_tor=np.array([20000., 20000., 12000.]),
    )

    ctrl = DSLPIDControl(env, pid_coeff=pid)

    rpms = np.array([14300, 14300, 14300, 14300])

    target_pos = np.zeros(3)
    target_vel = np.zeros(3)
    target_acc = np.zeros(3)
    target_rpy = np.zeros(3)
    target_rpy_rates = np.zeros(3)
    print(target_pos, target_vel, target_acc, target_rpy, target_rpy_rates)

    # Initial target position
    pos = np.array([0, 0, 1.0])
    print(pos)

    s_target_x = p.addUserDebugParameter("target_x", -2, 2, pos[0])
    s_target_y = p.addUserDebugParameter("target_y", -2, 2, pos[1])
    s_target_z = p.addUserDebugParameter("target_z", 0, 4, pos[2])


    def get_gui_values():
        tg_x = p.readUserDebugParameter(int(s_target_x))
        tg_y = p.readUserDebugParameter(int(s_target_y))
        tg_z = p.readUserDebugParameter(int(s_target_z))
        return tg_x, tg_y, tg_z


    tg_vel = np.zeros(3)
    tg_rpy = np.zeros(3)
    tg_rpy_rates = np.zeros(3)

    d_log = DroneDataLogger(
        num_drones=1,
        logging_freq=env.get_sim_freq(),
        logging_duration=0,
    )

    step_num = 1_000
    for i in range(step_num):
        kis = env.step(rpms)

        tg_x, tg_y, tg_z = get_gui_values()
        rpms, _, _ = ctrl.compute_control(
            control_timestep=env.get_sim_time_step(),
            current_position=kis[0].pos,
            current_quaternion=kis[0].quat,
            current_velocity=kis[0].vel,
            current_ang_velocity=kis[0].ang_vel,
            target_position=np.array([tg_x, tg_y, tg_z]),
            target_velocity=tg_vel,
            target_rpy=tg_rpy,
            target_rpy_rates=tg_rpy_rates,
        )

        t_stamp = i / env.get_sim_freq()
        d_log.log(
            drone_id=0,
            time_stamp=t_stamp,
            kin_state=kis[0],
        )

        time.sleep(env.get_sim_time_step())

    d_log.plot()
    env.close()
