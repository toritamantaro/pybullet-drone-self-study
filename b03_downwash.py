import numpy as np

from util.data_definition import DroneForcePIDCoefficients
from util.data_definition import DroneType, PhysicsType
from util.data_definition import DroneControlTarget

from blt_env.drone import DroneBltEnv
from control.drone_ctrl import DSLPIDControl

# Logger class to store drone status (optional).
from util.data_logger import DroneDataLogger

if __name__ == "__main__":

    # Initialize the PyBullet simulation environment.
    init_xyzs = np.array([[.5, 0, 1], [-.5, 0, .5]])
    aggr_phy_steps = 5
    num_drone = 2
    urdf_file = './assets/drone_x_01.urdf'
    drone_type = DroneType.QUAD_X
    phy_mode = PhysicsType.PYB_DW

    blt_env = DroneBltEnv(
        num_drones=num_drone,
        urdf_path=urdf_file,
        d_type=drone_type,
        phy_mode=phy_mode,
        is_gui=True,
        aggr_phy_steps=aggr_phy_steps,
        init_xyzs=init_xyzs,
        is_real_time_sim=True,
    )

    # Initialize the simulation aviary.
    eps_sec = 20
    period = 5
    action_freq = blt_env.get_sim_freq() / blt_env.get_aggr_phy_steps()
    num_wp = int(action_freq * period)
    target_pos = np.zeros((num_drone, num_wp, 3))
    for i in range(num_wp):
        target_pos[0, i, :] = [
            0.5 * np.cos(2 * np.pi * (i / num_wp)),
            0.5 * np.sin(2 * np.pi * (i / num_wp)),
            init_xyzs[0, 2] + (np.cos(4 * np.pi * (i / num_wp)) - 1) * 0.25,
        ]
        target_pos[1, i, :] = [
            0.5 * np.cos(2 * np.pi * (i / num_wp)),
            -0.5 * np.sin(2 * np.pi * (i / num_wp)),
            init_xyzs[1, 2] + (1 - np.cos(4 * np.pi * (i / num_wp))) * 0.25,
        ]

    wp_counters = np.array([0, int(num_wp / 2)])

    # Initialize the logger (optional).
    d_log = DroneDataLogger(
        num_drones=num_drone,
        logging_freq=int(action_freq),
        logging_duration=eps_sec,
    )

    # Initialize PID controllers.
    init_pid = DroneForcePIDCoefficients(
        P_for=np.array([.4, .4, 1.25], dtype=np.float32),
        I_for=np.array([.05, .05, .05], dtype=np.float32),
        D_for=np.array([.2, .2, .5], dtype=np.float32),
        P_tor=np.array([70000., 70000., 60000.], dtype=np.float32),
        I_tor=np.array([.0, .0, 500.], dtype=np.float32),
        D_tor=np.array([20000., 20000., 12000.], dtype=np.float32),
    )

    ctrls = [
        DSLPIDControl(
            env=blt_env,
            pid_coeff=init_pid,
        ) for _ in range(num_drone)
    ]

    # Run the simulation
    ctrl_event_n_steps = aggr_phy_steps
    action = np.array([np.array([0, 0, 0, 0]) for i in range(num_drone)])

    for i in range(0, int(eps_sec * blt_env.get_sim_freq()), aggr_phy_steps):
        # Step the simulation.
        kis = blt_env.step(action)

        # Compute control at the desired frequency.
        if i % ctrl_event_n_steps == 0:
            # Compute control for the current way point.
            for j in range(num_drone):
                action[j], _, _ = ctrls[j].compute_control_from_kinematics(
                    control_timestep=ctrl_event_n_steps * blt_env.get_sim_time_step(),
                    kin_state=kis[j],
                    ctrl_target=DroneControlTarget(
                        pos=target_pos[j, wp_counters[j], :],
                    ),
                )

            # Go to the next way point and loop
            for j in range(num_drone):
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (num_wp - 1) else 0

        # Log the simulation (optional).
        rpms = blt_env.get_last_rpm_values()
        for j in range(num_drone):
            d_log.log(
                drone_id=j,
                time_stamp=i / (blt_env.get_sim_freq()),
                kin_state=kis[j],
                rpm_values=rpms[j],
            )

    # Close the environment
    blt_env.close()

    # Plot the simulation results (optional).
    d_log.plot()



