import numpy as np
import pybullet as p

from util.data_definition import DroneType, PhysicsType
from util.data_definition import DroneForcePIDCoefficients, DroneControlTarget
from blt_env.drone import DroneBltEnv

from control.drone_ctrl import DSLPIDControl

# # Logger class to store drone status (optional).
# from util.data_logger import DroneDataLogger

if __name__ == "__main__":

    urdf_file = './assets/drone_x_01.urdf'
    drone_type = DroneType.QUAD_X
    phy_mode = PhysicsType.PYB_DW

    init_xyzs = np.array([[0, 0, 1.5]])

    env = DroneBltEnv(
        urdf_path=urdf_file,
        d_type=drone_type,
        is_gui=True,
        phy_mode=phy_mode,
        is_real_time_sim=True,
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

    # Initial target position
    pos = np.array([0, 0, 1.0])

    s_target_x = p.addUserDebugParameter("target_x", -2, 2, pos[0])
    s_target_y = p.addUserDebugParameter("target_y", -2, 2, pos[1])
    s_target_z = p.addUserDebugParameter("target_z", 0, 4, pos[2])

    def get_gui_values():
        tg_x = p.readUserDebugParameter(int(s_target_x))
        tg_y = p.readUserDebugParameter(int(s_target_y))
        tg_z = p.readUserDebugParameter(int(s_target_z))
        return tg_x, tg_y, tg_z

    # # Initialize the logger (optional).
    # d_log = DroneDataLogger(
    #     num_drones=1,
    #     logging_freq=int(env.get_sim_freq()),
    # )

    step_num = 2_000
    for i in range(step_num):
        kis = env.step(rpms)

        tg_x, tg_y, tg_z = get_gui_values()

        rpms, _, _ = ctrl.compute_control_from_kinematics(
            control_timestep=env.get_sim_time_step(),
            kin_state=kis[0],
            ctrl_target=DroneControlTarget(
                pos=np.array([tg_x, tg_y, tg_z]),
            ),
        )

        # # Log the simulation (optional).
        # t_stamp = i / env.get_sim_freq()
        # d_log.log(
        #     drone_id=0,
        #     time_stamp=t_stamp,
        #     kin_state=kis[0],
        # )

    # Close the environment
    env.close()

    # # Plot the simulation results (optional).
    # d_log.plot()
