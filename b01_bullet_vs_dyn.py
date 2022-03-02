
import time

import numpy as np
import pybullet as p

from util.data_definition import DroneType, PhysicsType
from blt_env.drone import DroneBltEnv

# Logger class to store drone status (optional).
from util.data_logger import DroneDataLogger


if __name__ == "__main__":

    urdf_file = './assets/drone_p_01.urdf'
    drone_type = DroneType.QUAD_PLUS

    ## Select a pysical model.
    phy_mode = PhysicsType.PYB  # Dynamics computing by pybullet.
    # phy_mode = PhysicsType.DYN  # Dynamics computing by explicit method.

    init_xyzx = np.array([[0, 0, 1.5]])

    env = DroneBltEnv(
        urdf_path=urdf_file,
        d_type=drone_type,
        phy_mode=phy_mode,
        init_xyzs=init_xyzx,
    )

    sim_freq = env.get_sim_freq()
    dp = env.get_drone_properties()
    max_rpm = dp.max_rpm
    hover_rpm = dp.hover_rpm

    rpm = hover_rpm * np.ones(4)

    sld_r0 = p.addUserDebugParameter(f"rotor 0 rpm", 0, max_rpm, hover_rpm)
    sld_r1 = p.addUserDebugParameter(f"rotor 1 rpm", 0, max_rpm, hover_rpm)
    sld_r2 = p.addUserDebugParameter(f"rotor 2 rpm", 0, max_rpm, hover_rpm)
    sld_r3 = p.addUserDebugParameter(f"rotor 3 rpm", 0, max_rpm, hover_rpm)


    def get_gui_values():
        r0 = p.readUserDebugParameter(int(sld_r0))
        r1 = p.readUserDebugParameter(int(sld_r1))
        r2 = p.readUserDebugParameter(int(sld_r2))
        r3 = p.readUserDebugParameter(int(sld_r3))
        return r0, r1, r2, r3


    # # Logger to store drone status (optional).
    # d_log = DroneDataLogger(num_drones=1, logging_freq=sim_freq, logging_duration=0, )

    step_num = 240 * 3
    for i in range(step_num):
        ki = env.step(rpm)

        rpm = np.array(get_gui_values())

        # # Logger to store drone status (optional).
        # d_log.log(drone_id=0, time_stamp=(i / sim_freq), kin_state=ki[0], rpm_values=rpm)

        time.sleep(1 / sim_freq)

    env.close()

    # # Logger to store drone status (optional).
    # d_log.plot()

