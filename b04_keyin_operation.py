import numpy as np

from util.data_definition import DroneForcePIDCoefficients
from util.data_definition import DroneType, PhysicsType
from util.data_definition import DroneControlTarget
from blt_env.drone import DroneBltEnv
from control.drone_ctrl import DSLPIDControl
from util.external_input import KeyboardInputCtrl
from dev.bullet_cam import BulletCameraDevice, compute_view_matrix_from_cam_location
from util.data_logger import DroneDataLogger

if __name__ == "__main__":

    # Initialize the PyBullet simulation environment.
    init_xyzx = np.array([[.5, 0, 1], [-.5, 0, .5]])
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
        init_xyzs=init_xyzx,
        is_real_time_sim=True,
    )

    # Initialize the camera.
    my_camera = BulletCameraDevice(
        res_w=640,
        res_h=480,
        z_near=0.01,
        z_far=10.0,
        fov_w=50,
    )

    # Initialize the simulation aviary.
    eps_sec = 30
    action_freq = blt_env.get_sim_freq() / blt_env.get_aggr_phy_steps()

    # Initialize the logger (optional).
    d_log = DroneDataLogger(
        num_drones=num_drone,
        logging_freq=int(action_freq),
        logging_duration=eps_sec,
    )

    # Initialize PID controllers.
    init_pid = DroneForcePIDCoefficients(
        P_for=np.array([.4, .4, 1.25]),
        I_for=np.array([.05, .05, .05]),
        D_for=np.array([.2, .2, .5]),
        P_tor=np.array([70000., 70000., 60000.]),
        I_tor=np.array([.0, .0, 500.]),
        D_tor=np.array([20000., 20000., 12000.]),
    )

    ctrls = [
        DSLPIDControl(
            env=blt_env,
            pid_coeff=init_pid,
        ) for _ in range(num_drone)
    ]

    # Run the simulation.
    ctrl_event_n_steps = aggr_phy_steps
    action = np.array([np.array([0, 0, 0, 0]) for i in range(num_drone)])

    # Initialize the keyboard controller.
    target_drone_id = 0
    key_ctrl = KeyboardInputCtrl(blt_env=blt_env, nth_drone=target_drone_id)

    for i in range(0, int(eps_sec * blt_env.get_sim_freq()), aggr_phy_steps):
        # Step the simulation
        kis = blt_env.step(action)

        # Compute control at the desired frequency
        if i % ctrl_event_n_steps == 0:
            # Compute control target from keyin.
            ctrl_target = key_ctrl.get_ctrl_target()

            for j in range(num_drone):
                ki = kis[j]
                if j != target_drone_id:
                    ctrl_target = DroneControlTarget(
                        pos=np.array(init_xyzx[j]),
                        vel=np.zeros(3),
                        rpy=np.zeros(3),
                    )

                action[j], _, _ = ctrls[j].compute_control_from_kinematics(
                    control_timestep=ctrl_event_n_steps * blt_env.get_sim_time_step(),
                    kin_state=ki,
                    ctrl_target=ctrl_target,
                )

            # Cam capture.
            cam_pos = kis[0].pos + np.array([0, 0, 0.02])
            cam_quat = kis[0].quat
            view_mat = compute_view_matrix_from_cam_location(cam_pos=cam_pos, cam_quat=cam_quat, )
            _, _, _ = my_camera.cam_capture(view_mat)

        # Log the simulation (optional).
        rpms = blt_env.get_last_rpm_values()
        for j in range(num_drone):
            d_log.log(
                drone_id=j,
                time_stamp=i / (blt_env.get_sim_freq()),
                kin_state=kis[j],
                rpm_values=rpms[j],
            )

    # Close the environment.
    blt_env.close()

    # Plot the simulation results (optional).
    d_log.plot()
