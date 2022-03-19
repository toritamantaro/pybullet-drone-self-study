

from cycler import cycler  # color cycle
import numpy as np
import matplotlib.pyplot as plt

from util.data_definition import DroneKinematicsInfo, DroneControlTarget


class DroneDataLogger(object):
    """
    16 states: pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,roll,pitch,yaw,ang_vel_x,ang_vel_y,ang_vel_z,rpm0,rpm1, rpm2, rpm3
    12 control targets: pos_x,pos_y,pos_z,vel_x, vel_y,vel_z,roll,pitch,yaw,ang_vel_x,ang_vel_y,ang_vel_z
    """

    def __init__(self,
                 num_drones: int,
                 logging_freq: int,  # logging frequency [Hz]
                 logging_duration: float = 0  # logging duration time [sec]
                 ):
        self._logging_freq = logging_freq
        self._logging_duration = logging_duration
        self._pre_allocated_arrays = False if self._logging_duration == 0 else True
        self._num_drones = num_drones
        self._counters = np.zeros(self._num_drones)
        step_num = int(self._logging_duration * self._logging_freq)
        self._time_stamps = np.zeros((self._num_drones, step_num))
        self._counters = np.zeros(self._num_drones)
        self._timestamps = np.zeros((self._num_drones, step_num))
        self._states = np.zeros((self._num_drones, 16, step_num))
        self._targets = np.zeros((self._num_drones, 12, step_num))

    @property
    def logging_freq(self) -> int:
        return self._logging_freq

    @property
    def logging_duration(self) -> float:
        return self._logging_duration

    def log_numpy(self,
                  drone_id: int,  # Id of the drone associated to the log entry.
                  time_stamp: float,  # Timestamp of the log in simulation clock.
                  state_and_rpm: np.ndarray,
                  target: np.ndarray = np.zeros(12),
                  ):
        cls_name = self.__class__.__name__
        assert (drone_id >= 0) and (drone_id <= self._num_drones - 1), f"{cls_name}:{drone_id}:Invalid drone id."
        assert time_stamp >= 0, f"{cls_name}:{time_stamp}: Invalid time_stamp."
        assert (isinstance(state_and_rpm, np.ndarray)) and (state_and_rpm.shape == (16,)), f"{cls_name}:{len(state_and_rpm)}:Invalid state."
        assert (isinstance(target, np.ndarray)) and (target.shape == (12,)), f"{cls_name}:{len(target)}:Invalid target."

        current_counter = int(self._counters[drone_id])
        # Add rows to the matrices if a counter exceeds their size
        if current_counter >= self._time_stamps.shape[1]:
            self._time_stamps = np.concatenate((self._time_stamps, np.zeros((self._num_drones, 1))), axis=1)
            self._states = np.concatenate((self._states, np.zeros((self._num_drones, 16, 1))), axis=2)
            self._targets = np.concatenate((self._targets, np.zeros((self._num_drones, 12, 1))), axis=2)
        # Advance a counter is the matrices have overgrown it
        elif not self._pre_allocated_arrays and self._time_stamps.shape[1] > current_counter:
            current_counter = self._time_stamps.shape[1] - 1
        # Log the information and increase the counter
        self._time_stamps[drone_id, current_counter] = time_stamp
        self._states[drone_id, :, current_counter] = state_and_rpm
        self._targets[drone_id, :, current_counter] = target
        self._counters[drone_id] = current_counter + 1

    def log(
            self,
            drone_id: int,  # Id of the drone associated to the log entry.
            time_stamp: float,  # Timestamp of the log in simulation clock.
            kin_state: DroneKinematicsInfo,
            rpm_values: np.ndarray = np.zeros(4),
            ctrl_target: DroneControlTarget = DroneControlTarget(),
    ):
        """
        Parameters
        ----------
        drone_id
        time_stamp
        kin_state
        rpm_values : 4 rotors rpm
        ctrl_target : If not specified, dummy target data will be stored.
        """

        state_np = np.array([
            kin_state.pos[0],  # pos_x
            kin_state.pos[1],  # pos_y
            kin_state.pos[2],  # pos_z
            kin_state.vel[0],  # vel_x
            kin_state.vel[1],  # vel_y
            kin_state.vel[2],  # vel_z
            kin_state.rpy[0],  # roll
            kin_state.rpy[1],  # pitch
            kin_state.rpy[2],  # yaw
            kin_state.ang_vel[0],  # ang_vel_x
            kin_state.ang_vel[1],  # ang_vel_y
            kin_state.ang_vel[2],  # ang_vel_z
            rpm_values[0],  # rotor_0 rpm
            rpm_values[1],  # rotor_1 rpm
            rpm_values[2],  # rotor_2 rpm
            rpm_values[3],  # rotor_3 rpm
        ])

        target_np = np.array([
            ctrl_target.pos[0],  # pos_x
            ctrl_target.pos[1],  # pos_y
            ctrl_target.pos[2],  # pos_z
            ctrl_target.vel[0],  # vel_x
            ctrl_target.vel[1],  # vel_y
            ctrl_target.vel[2],  # vel_z
            ctrl_target.rpy[0],  # roll
            ctrl_target.rpy[1],  # pitch
            ctrl_target.rpy[2],  # yaw
            ctrl_target.rpy_rates[0],  # rpy_rates_x
            ctrl_target.rpy_rates[1],  # rpy_rates_y
            ctrl_target.rpy_rates[2],  # rpy_rates_z
        ])

        self.log_numpy(
            drone_id=drone_id,
            time_stamp=time_stamp,
            state_and_rpm=state_np,
            target=target_np,
        )

    def plot(self, pwm=False):
        """Logs entries for a single simulation step, of a single drone.

        This is a reference from the following ...

            https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/utils/Logger.py

        Parameters
        ----------
        pwm : bool, optional
            If True, converts logged RPM into PWM values (for Crazyflies).

        """
        # Loop over colors and line styles
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))
        fig, axs = plt.subplots(10, 2)
        t = np.arange(0, self._time_stamps.shape[1] / self._logging_freq, 1 / self._logging_freq)
        t = t[:self._time_stamps.shape[1]]  # keep the lengths even

        # Column
        col = 0

        # XYZ
        row = 0
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 0, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')

        row = 1
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 1, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')

        row = 2
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 2, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')

        # RPY
        row = 3
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 6, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r (rad)')
        row = 4
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 7, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('p (rad)')
        row = 5
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 8, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (rad)')

        # Ang Vel
        row = 6
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 9, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wx')
        row = 7
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 10, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wy')
        row = 8
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 11, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wz')

        # Time
        row = 9
        axs[row, col].plot(t, t, label="time")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('time')

        # Column
        col = 1

        # Velocity
        row = 0
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 3, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')
        row = 1
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 4, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')
        row = 2
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 5, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')

        # RPY Rates
        row = 3
        for j in range(self._num_drones):
            rdot = np.hstack([0, (self._states[j, 6, 1:] - self._states[j, 6, 0:-1]) * self._logging_freq])
            axs[row, col].plot(t, rdot, label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('rdot (rad/s)')
        row = 4
        for j in range(self._num_drones):
            pdot = np.hstack([0, (self._states[j, 7, 1:] - self._states[j, 7, 0:-1]) * self._logging_freq])
            axs[row, col].plot(t, pdot, label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('pdot (rad/s)')
        row = 5
        for j in range(self._num_drones):
            ydot = np.hstack([0, (self._states[j, 8, 1:] - self._states[j, 8, 0:-1]) * self._logging_freq])
            axs[row, col].plot(t, ydot, label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('ydot (rad/s)')

        # This IF converts RPM into PWM for all drones
        for j in range(self._num_drones):
            for i in range(12, 16):
                if pwm and j > 0:
                    self._states[j, i, :] = (self._states[j, i, :] - 4070.3) / 0.2685

        # RPMs
        row = 6
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 12, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM0')
        else:
            axs[row, col].set_ylabel('RPM0')
        row = 7
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 13, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM1')
        else:
            axs[row, col].set_ylabel('RPM1')
        row = 8
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 14, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM2')
        else:
            axs[row, col].set_ylabel('RPM2')
        row = 9
        for j in range(self._num_drones):
            axs[row, col].plot(t, self._states[j, 15, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM3')
        else:
            axs[row, col].set_ylabel('RPM3')

        # Drawing options
        for i in range(10):
            for j in range(2):
                axs[i, j].grid(True)
                axs[i, j].legend(loc='upper right',
                                 frameon=True
                                 )
        fig.subplots_adjust(left=0.06,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.15,
                            hspace=0.0
                            )
        plt.show()
