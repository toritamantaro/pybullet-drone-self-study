from typing import Optional, List, Tuple, Union
import time
import random

from logging import getLogger, NullHandler

import numpy as np
import pybullet as p
import pybullet_data

from blt_env.bullet_base import BulletEnv

from util.data_definition import DroneProperties, DroneType, DroneKinematicsInfo, PhysicsType
from util.file_tools import DroneUrdfAnalyzer

logger = getLogger(__name__)
logger.addHandler(NullHandler())


# logger.setLevel(DEBUG)  # for standalone debugging
# logger.addHandler(StreamHandler())  # for standalone debugging

def real_time_step_synchronization(sim_counts, start_time, time_step):
    """Syncs the stepped simulation with the wall-clock.

    This is a reference from the following ...

        https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/utils/utils.py

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    sim_counts : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    time_step : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if time_step > .04 or sim_counts % (int(1 / (24 * time_step))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (sim_counts * time_step):
            time.sleep(time_step * sim_counts - elapsed)


def load_drone_properties(file_path: str, d_type: DroneType) -> DroneProperties:
    file_analyzer = DroneUrdfAnalyzer()
    return file_analyzer.parse(file_path, int(d_type))


class DroneBltEnv(BulletEnv):

    def __init__(
            self,
            urdf_path: str,
            d_type: DroneType = DroneType.QUAD_PLUS,
            phy_mode: PhysicsType = PhysicsType.PYB,
            sim_freq: int = 240,
            aggr_phy_steps: int = 1,
            num_drones: int = 1,
            is_gui: bool = True,
            is_real_time_sim: bool = False,
            init_xyzs: Optional[Union[List, np.ndarray]] = None,
            init_rpys: Optional[Union[List, np.ndarray]] = None,
    ):
        """
        'aggr_phy_steps'は、self.step(rpm_values: np.ndarray)が１回よばれた際に、PyBulletによるシミュレーションを
        何ステップ実行するかを指定する。この値が増加すると、actionの頻度は減少する。
        （この場合のactionとは、rpm_valuesをself.step()に引数として与えてドローンをコントロールすること）

        Parameters
        ----------
        urdf_path : The drone *.URDF file path.
        d_type : Specifies the type of drone to be loaded from the *.URDF file.
        phy_mode : Specifies the type of physics simulation for PyBullet.
        sim_freq : Specifies the frequency of the PyBullet step simulations.
        aggr_phy_steps : The number of physics steps within one call to `self.step()`.
                        The frequency of the control action is changed by the aggr_phy_steps.
        num_drones : Number of drones to be loaded.
        is_gui : Whether to start PyBullet in GUI mode.
        """
        super().__init__(is_gui=is_gui)
        self._drone_type = d_type
        self._urdf_path = urdf_path
        self._physics_mode = phy_mode

        self._dp = load_drone_properties(self._urdf_path, self._drone_type)
        # self.printout_drone_properties()

        # PyBullet simulation settings.
        self._num_drones = num_drones
        self._aggr_phy_steps = aggr_phy_steps
        self._g = self._dp.g
        self._sim_freq = sim_freq
        self._sim_time_step = 1. / self._sim_freq
        self._is_realtime_sim = is_real_time_sim  # add wait time in step().

        # Initialization position of the drones.
        if init_xyzs is None:
            self._init_xyzs = np.vstack([
                np.array([x * 4 * self._dp.l for x in range(self._num_drones)]),
                np.array([y * 4 * self._dp.l for y in range(self._num_drones)]),
                np.ones(self._num_drones) * (self._dp.collision_h / 2 - self._dp.collision_z_offset + 0.1),
            ]).transpose().reshape(self._num_drones, 3)
        else:
            assert init_xyzs.ndim == 2, f"'init_xyzs' should has 2 dimension. current dims are {init_xyzs.ndim}."
            self._init_xyzs = np.array(init_xyzs)
        assert self._init_xyzs.shape[0] == self._num_drones, f""" Initialize position error.
        Number of init pos {self._init_xyzs.shape[0]} vs number of drones {self._num_drones}."""

        if init_rpys is None:
            self._init_rpys = np.zeros((self._num_drones, 3))
        else:
            assert init_rpys.ndim == 2, f"'init_rpys' should has 2 dimension. current dims are {init_rpys.ndim}."
            self._init_rpys = np.array(init_rpys)
        assert self._init_rpys.shape[0] == self._num_drones, f""" Initialize roll, pitch and yaw error.
        Number of init rpy {self._init_rpys.shape[0]} vs number of drones {self._num_drones}."""

        # Simulation status.
        self._sim_counts = 0
        self._last_rpm_values = np.zeros((self._num_drones, 4))
        ''' 
        The 'DroneKinematicInfo' class is simply a placeholder for the following information.
            pos : position
            quat : quaternion
            rpy : roll, pitch and yaw
            vel : linear velocity
            ang_vel : angular velocity
        '''
        self._kis = [DroneKinematicsInfo() for _ in range(self._num_drones)]

        if self._physics_mode == PhysicsType.DYN:
            self._rpy_rates = np.zeros((self._num_drones, 3))

        # PyBullet environment.
        self._client = p.connect(p.GUI) if self._is_gui else p.connect(p.DIRECT)
        p.setGravity(0, 0, -self._g, physicsClientId=self._client)
        p.setRealTimeSimulation(0, physicsClientId=self._client)
        p.setTimeStep(self._sim_time_step, physicsClientId=self._client)

        # Load objects.
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF('plane.urdf')

        # Load drones.
        self._drone_ids = np.array([
            p.loadURDF(
                self._urdf_path,
                self._init_xyzs[i, :],
                p.getQuaternionFromEuler(self._init_rpys[i, :]),
            ) for i in range(self._num_drones)])

        # Update the information before running the simulations.
        self.update_drones_kinematic_info()

        # Start measuring time.
        self._start_time = time.time()

    def get_sim_time_step(self) -> float:
        return self._sim_time_step

    def get_sim_counts(self) -> int:
        return self._sim_counts

    def get_drone_properties(self) -> DroneProperties:
        return self._dp

    def get_drones_kinematic_info(self) -> List[DroneKinematicsInfo]:
        return self._kis

    def get_aggr_phy_steps(self) -> int:
        return self._aggr_phy_steps

    def get_sim_freq(self) -> int:
        return self._sim_freq

    def get_num_drones(self) -> int:
        return self._num_drones

    def get_last_rpm_values(self) -> np.ndarray:
        return self._last_rpm_values

    def refresh_bullet_env(self):
        """
        Refresh the PyBullet simulation environment.
        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `self.reset()` function.

        PyBulletのシミュレーション環境を初期化する

        """
        self._sim_counts = 0
        self._last_rpm_values = np.zeros((self._num_drones, 4))
        self._kis = [DroneKinematicsInfo() for _ in range(self._num_drones)]
        if self._physics_mode == PhysicsType.DYN:
            self._rpy_rates = np.zeros((self._num_drones, 3))

        # Set PyBullet's parameters.
        p.setGravity(0, 0, -self._g, physicsClientId=self._client)
        p.setRealTimeSimulation(0, physicsClientId=self._client)
        p.setTimeStep(self._sim_time_step, physicsClientId=self._client)

        # Load objects.
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        self._plane_id = p.loadURDF('plane.urdf')

        # Load drones.
        self._drone_ids = np.array([
            p.loadURDF(
                self._urdf_path,
                self._init_xyzs[i, :],
                p.getQuaternionFromEuler(self._init_rpys[i, :]),
            ) for i in range(self._num_drones)])

        self.update_drones_kinematic_info()

        # Reset measuring time.
        self._start_time = time.time()

    def update_drones_kinematic_info(self):
        for i in range(self._num_drones):
            pos, quat = p.getBasePositionAndOrientation(
                bodyUniqueId=self._drone_ids[i],
                physicsClientId=self._client,
            )
            rpy = p.getEulerFromQuaternion(quat)
            vel, ang_vel = p.getBaseVelocity(
                bodyUniqueId=self._drone_ids[i],
                physicsClientId=self._client,
            )
            self._kis[i] = DroneKinematicsInfo(
                pos=np.array(pos),
                quat=np.array(quat),
                rpy=np.array(rpy),
                vel=np.array(vel),
                ang_vel=np.array(ang_vel),
            )

    def close(self) -> None:
        if p.isConnected() != 0:
            p.disconnect(physicsClientId=self._client)

    def reset(self) -> None:
        if p.isConnected() != 0:
            p.resetSimulation(physicsClientId=self._client)
            self.refresh_bullet_env()

    def step(self, rpm_values: np.ndarray) -> List[DroneKinematicsInfo]:
        """
        Parameters
        ----------
        rpm_values : Multiple arrays with 4 values as a pair of element.
                    Specify the rotational speed of the four rotors of each drone.
        """
        rpm_values = self.check_values_for_rotors(rpm_values)

        for _ in range(self._aggr_phy_steps):
            '''
            Update and store the drones kinematic info the same action value of "rpm_values" 
            for the number of times specified by "self._aggr_phy_steps".
            '''
            if self._aggr_phy_steps > 1 and self._physics_mode in [
                PhysicsType.DYN,
                PhysicsType.PYB_GND,
                PhysicsType.PYB_DRAG,
                PhysicsType.PYB_DW,
                PhysicsType.PYB_GND_DRAG_DW
            ]:
                self.update_drones_kinematic_info()

            # step the simulation
            for i in range(self._num_drones):
                self.physics(
                    rpm_values[i, :],
                    i,
                    self._last_rpm_values[i, :],
                )

            # In the case of the explicit solution technique, 'p.stepSimulation()' is not used.
            if self._physics_mode != PhysicsType.DYN:
                p.stepSimulation(physicsClientId=self._client)

            # Save the last applied action (for compute e.g. drag)
            self._last_rpm_values = rpm_values

        # Update and store the drones kinematic information
        self.update_drones_kinematic_info()

        # Advance the step counter
        self._sim_counts = self._sim_counts + (1 * self._aggr_phy_steps)

        # Synchronize the step interval with real time.
        if self._is_realtime_sim:
            real_time_step_synchronization(self._sim_counts, self._start_time, self._sim_time_step)

        return self._kis

    def check_values_for_rotors(self, rpm_values: np.ndarray) -> np.ndarray:
        """
        Check that 'rpm_values', which specifies the rotation speed of the 4-rotors, are in the proper form.
        Also, if possible, modify 'rpm_values' to the appropriate form.

        各ドローンの４つのロータ回転数を指定するnp.ndarrayが、適切な形式になっているか確認する

        Parameters and Returns
        ----------
        rpm_values : Multiple arrays with 4 values as a pair of element.
                    Specify the rotational speed of the four rotors of each drone.
        """
        cls_name = self.__class__.__name__
        assert isinstance(rpm_values, np.ndarray), f"Invalid rpm_values type is used on {cls_name}."
        assert rpm_values.ndim == 1 or rpm_values.ndim == 2, f"Invalid dimension of rpm_values is used on {cls_name}."
        if rpm_values.ndim == 1:
            assert len(rpm_values) == 4, f"Invalid number of elements were used for rpm_values on {cls_name}."
            ''' e.g.
            while, a = [100, 200, 300, 400]
            then, np.tile(a, (3, 1)) -> [[100, 200, 300, 400], [100, 200, 300, 400], [100, 200, 300, 400]]
            '''
            rpm_values = np.tile(rpm_values, (self._num_drones, 1))
        elif rpm_values.ndim == 2:
            assert rpm_values.shape[1] == 4, f"Invalid number of elements were used for rpm_values on {cls_name}."
            rpm_values = np.reshape(rpm_values, (self._num_drones, 4))
        return rpm_values

    def physics(
            self,
            rpm: np.ndarray,
            nth_drone: int,
            last_rpm: Optional[np.ndarray],
    ) -> None:
        """
        The type of physics simulation will be selected according to 'self._physics_mode'.

        'self._physics_mode'で指定されたモードにしたがって物理演算モデルが選択される

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        last_rpm : Previous specified value.
        """

        def pyb(rpm, nth_drone: int, last_rpm=None):
            self.apply_rotor_physics(rpm, nth_drone)

        def dyn(rpm, nth_drone: int, last_rpm=None):
            self.apply_dynamics(rpm, nth_drone)

        def pyb_gnd(rpm, nth_drone: int, last_rpm=None):
            self.apply_rotor_physics(rpm, nth_drone)
            self.apply_ground_effect(rpm, nth_drone)

        def pyb_drag(rpm, nth_drone: int, last_rpm):
            self.apply_rotor_physics(rpm, nth_drone)
            self.apply_drag(last_rpm, nth_drone)  # apply last data

        def pyb_dw(rpm, nth_drone: int, last_rpm=None):
            self.apply_rotor_physics(rpm, nth_drone)
            self.apply_downwash(nth_drone)

        def pyb_gnd_drag_dw(rpm, nth_drone: int, last_rpm):
            self.apply_rotor_physics(rpm, nth_drone)
            self.apply_ground_effect(rpm, nth_drone)
            self.apply_drag(last_rpm, nth_drone)  # apply last data
            self.apply_downwash(nth_drone)

        def other(rpm, nth_drone: int, last_rpm):
            logger.error(f"In {self.__class__.__name__}, invalid physic mode key.")

        phy_key = self._physics_mode.value

        key_dict = {
            'pyb': pyb,
            'dyn': dyn,
            'pyb_gnd': pyb_gnd,
            'pyb_drag': pyb_drag,
            'pyb_dw': pyb_dw,
            'pyb_gnd_drag_dw': pyb_gnd_drag_dw,
        }
        return key_dict.get(phy_key, other)(rpm, nth_drone, last_rpm)

    def apply_rotor_physics(self, rpm: np.ndarray, nth_drone: int):
        """
        Apply the individual thrusts and torques generated by the motion of the four rotors.
        4つのローターの動きによって発生する個々の推力とトルクを単純に適用

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        """
        assert len(rpm) == 4, f"The length of rpm_values must be 4. currently it is {len(rpm)}."
        forces = (np.array(rpm) ** 2) * self._dp.kf
        torques = (np.array(rpm) ** 2) * self._dp.km
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(
                objectUniqueId=self._drone_ids[nth_drone],
                linkIndex=i,  # link id of the rotors.
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self._client,
            )
        p.applyExternalTorque(
            objectUniqueId=self._drone_ids[nth_drone],
            linkIndex=4,  # link id of the center of mass.
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self._client,
        )

    def apply_ground_effect(self, rpm: np.ndarray, nth_drone: int):
        """
        Apply ground effect.
        地面効果を適用

        This is a reference from the following ...

            https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py

            Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        """
        assert len(rpm) == 4, f"The length of rpm_values must be 4. currently it is {len(rpm)}."

        ''' getLinkState()
        computeLinkVelocity : 
            If set to 1, the Cartesian world velocity will be computed and returned.
        computeForwardKinematics : 
            If set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
        '''
        link_states = np.array(
            p.getLinkStates(
                bodyUniqueId=self._drone_ids[nth_drone],
                linkIndices=[0, 1, 2, 3, 4],
                computeLinkVelocity=1,
                computeForwardKinematics=1,
                physicsClientId=self._client,
            ),
            dtype=object,
        )

        # Simple, per-propeller ground effects.
        prop_heights = np.array(
            [link_states[0, 0][2], link_states[1, 0][2], link_states[2, 0][2], link_states[3, 0][2]])
        prop_heights = np.clip(prop_heights, self._dp.grand_eff_h_clip, np.inf)
        gnd_effects = np.array(rpm) ** 2 * self._dp.kf * self._dp.gnd_eff_coeff * (
                self._dp.prop_radius / (4 * prop_heights)) ** 2

        ki = self._kis[nth_drone]
        if np.abs(ki.rpy[0]) < np.pi / 2 and np.abs(ki.rpy[1]) < np.pi / 2:
            for i in range(4):
                p.applyExternalForce(
                    objectUniqueId=self._drone_ids[nth_drone],
                    linkIndex=i,
                    forceObj=[0, 0, gnd_effects[i]],
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                    physicsClientId=self._client,
                )

    def apply_drag(self, rpm: np.ndarray, nth_drone: int):
        """
        Apply drag force.
        抗力を適用

        This is a reference from the following ...

            https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py

            Based on the the system identification in (Forster, 2015).

            Chapter 4 Drag Coefficients
            http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        """

        # Rotation matrix of the base.
        ki = self._kis[nth_drone]
        base_rot = np.array(p.getMatrixFromQuaternion(ki.quat)).reshape(3, 3)
        # Simple draft model applied to the center of mass
        drag_factors = -1 * self._dp.drag_coeff * np.sum(2 * np.pi * np.array(rpm) / 60)
        drag = np.dot(base_rot, drag_factors * np.array(ki.vel))
        p.applyExternalForce(
            objectUniqueId=self._drone_ids[nth_drone],
            linkIndex=4,  # link id of the center of mass.
            forceObj=drag,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self._client,
        )

    def apply_downwash(self, nth_drone: int):
        """
        Apply downwash.
        ダウンウオッシュ（吹き下ろし）を適用

        The aerodynamic caused by the motion of the rotor blade's airfoil during the process of generating lift.
        Interactions between multiple drones.

        This is a reference from the following ...

            https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py

            Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Parameters
        ----------
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        """
        ki_d = self._kis[nth_drone]
        for i in range(self._num_drones):
            ki_i = self._kis[i]
            delta_z = ki_i.pos[2] - ki_d.pos[2]
            delta_xy = np.linalg.norm(np.array(ki_i.pos[0:2]) - np.array(ki_d.pos[0:2]))
            if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
                alpha = self._dp.dw_coeff_1 * (self._dp.prop_radius / (4 * delta_z)) ** 2
                beta = self._dp.dw_coeff_2 * delta_z + self._dp.dw_coeff_3
                downwash = [0, 0, -alpha * np.exp(-0.5 * (delta_xy / beta) ** 2)]
                p.applyExternalForce(
                    objectUniqueId=self._drone_ids[nth_drone],
                    linkIndex=4,  # link id of the center of mass.
                    forceObj=downwash,
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                    physicsClientId=self._client,
                )

    def apply_dynamics(self, rpm: np.ndarray, nth_drone: int):
        """
        Apply dynamics taking into account moment of inertia, etc. (not pybullet base)
        慣性モーメントなどを考慮した力学を陽解法を用いて適用

        This is a reference from the following ...

            https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py

            Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        """
        assert len(rpm) == 4, f"The length of rpm_values must be 4. currently it is {len(rpm)}."

        # Current state.
        ki = self._kis[nth_drone]
        pos = ki.pos
        quat = ki.quat
        rpy = ki.rpy
        vel = ki.vel
        rpy_rates = self._rpy_rates[nth_drone]  # angular velocity
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # Compute thrust and torques.
        thrust, x_torque, y_torque, z_torque = self.rpm2forces(rpm)
        thrust = np.array([0, 0, thrust])

        thrust_world_frame = np.dot(rotation, thrust)
        forces_world_frame = thrust_world_frame - np.array([0, 0, self._dp.gf])

        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self._dp.J, rpy_rates))
        rpy_rates_deriv = np.dot(self._dp.J_inv, torques)  # angular acceleration
        no_pybullet_dyn_accs = forces_world_frame / self._dp.m

        # Update state.
        vel = vel + self._sim_time_step * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self._sim_time_step * rpy_rates_deriv
        pos = pos + self._sim_time_step * vel
        rpy = rpy + self._sim_time_step * rpy_rates

        # Set PyBullet state
        p.resetBasePositionAndOrientation(
            bodyUniqueId=self._drone_ids[nth_drone],
            posObj=pos,
            ornObj=p.getQuaternionFromEuler(rpy),
            physicsClientId=self._client,
        )

        # Note: the base's velocity only stored and not used.
        p.resetBaseVelocity(
            objectUniqueId=self._drone_ids[nth_drone],
            linearVelocity=vel,
            angularVelocity=[-1, -1, -1],  # ang_vel not computed by DYN
            physicsClientId=self._client,
        )

        # Store the roll, pitch, yaw rates for the next step
        # ki.rpy_rates = rpy_rates
        self._rpy_rates[nth_drone] = rpy_rates

    def rpm2forces(self, rpm: np.ndarray) -> Tuple:
        """
        Compute thrust and x, y, z axis torque at specified rotor speed.

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.

        Returns
        -------
        (
            thrust,  # It is sum of the thrust of the 4 rotors.
            x_torque,  # It is the torque generated by the thrust of the rotors.
            y_torque,  # It is the torque generated by the thrust of the rotors.
            z_torque,  #  It is sum of the torque of the 4 rotors.
        )
        """
        forces = np.array(rpm) ** 2 * self._dp.kf
        thrust = np.sum(forces)
        z_torques = np.array(rpm) ** 2 * self._dp.km
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self._drone_type == DroneType.QUAD_X:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self._dp.l / np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self._dp.l / np.sqrt(2))
        elif self._drone_type in [DroneType.QUAD_PLUS, DroneType.OTHER]:
            x_torque = (forces[1] - forces[3]) * self._dp.l
            y_torque = (-forces[0] + forces[2]) * self._dp.l
        return thrust, x_torque, y_torque, z_torque

    def printout_drone_properties(self) -> None:
        mes = f"""
        {self.__class__.__name__} loaded parameters from the .urdf :
        {self._urdf_path}
        m: {self._dp.m}
        l: {self._dp.l}
        ixx: {self._dp.ixx}
        iyy: {self._dp.iyy}
        izz: {self._dp.izz}
        kf: {self._dp.kf}
        km: {self._dp.km}
        J: {self._dp.J}
        thrust2weight_ratio: {self._dp.thrust2weight_ratio}
        max_speed_kmh: {self._dp.max_speed_kmh}
        gnd_eff_coeff: {self._dp.gnd_eff_coeff}
        prop_radius: {self._dp.prop_radius}
        drag_coeff_xy: {self._dp.drag_coeff_xy}
        drag_z_coeff: {self._dp.drag_coeff_z}
        dw_coeff_1: {self._dp.dw_coeff_1}
        dw_coeff_2: {self._dp.dw_coeff_2}
        dw_coeff_3: {self._dp.dw_coeff_3}
        gf: {self._dp.gf}
        hover_rpm: {self._dp.hover_rpm}
        max_rpm: {self._dp.max_rpm}
        max_thrust: {self._dp.max_thrust}
        max_xy_torque: {self._dp.max_xy_torque}
        max_z_torque: {self._dp.max_z_torque}
        grand_eff_h_clip: {self._dp.grand_eff_h_clip}
        grand_eff_h_clip: {self._dp.grand_eff_h_clip}
        A: {self._dp.A}
        B_coeff: {self._dp.B_coeff}
        Mixer: {self._dp.Mixer}
        """
        logger.info(mes)


if __name__ == "__main__":
    '''
    If you want to run this module by itself, try the following.

       $ python -m blt_env.drone

    '''

    urdf_file = './assets/drone_x_01.urdf'
    drone_type = DroneType.QUAD_X
    phy_mode = PhysicsType.PYB
    # phy_mode = PhysicsType.DYN

    env = DroneBltEnv(
        urdf_path=urdf_file,
        d_type=drone_type,
        is_gui=True,
        phy_mode=phy_mode,
        num_drones=2,
    )

    env.printout_drone_properties()

    rpms = np.array([14600, 14600, 14600, 14600])

    step_num = 1_000
    for _ in range(step_num):
        ki = env.step(rpms)
        # print(ki)
        time.sleep(env.get_sim_time_step())

    env.close()
