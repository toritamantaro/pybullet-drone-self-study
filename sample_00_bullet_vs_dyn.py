import time
from util.data_tools import DroneType, PhysicsType

from env.drone_test import BulletSimTest

from util.data_logger import DroneDataLogger
import numpy as np

import pybullet as p

if __name__ == "__main__":

    urdf_file = './assets/drone_p_01.urdf'
    drone_type = DroneType.QUAD_PLUS

    ## Select a pysical model.
    phy_mode = PhysicsType.PYB  # Dynamics computing by pybullet.
    # phy_mode = PhysicsType.DYN  # Dynamics computing by explicit method.

    env = BulletSimTest(
        urdf_path=urdf_file,
        d_type=drone_type,
        phy_mode=phy_mode,
    )

    sim_freq = env.sim_freq
    dp = env.drone_properties
    max_rpm = dp.max_rpm
    hover_rpm = dp.hover_rpm

    rpms = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])

    print(rpms)
    print('max', max_rpm, hover_rpm)
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


    ## Logger to store drone status (optional).
    # d_log = DroneDataLogger(num_drones=1, freq_hz=sim_freq, duration_sec=0, )

    step_num = 1_000
    for i in range(step_num):
        ki = env.step(rpms)

        rpms = np.array(get_gui_values())

        ## Logger to store drone status (optional).
        # d_log.log(drone_id=0, time_stamp=(i / sim_freq), state=ki, )

        time.sleep(1 / sim_freq)

    ## Logger to store drone status (optional).
    # d_log.plot()

    env.close()
