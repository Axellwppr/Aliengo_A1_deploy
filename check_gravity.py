import os
import sys
from typing import Optional
import time
import datetime
import numpy as np
import math
import itertools
from setproctitle import setproctitle
from scipy.spatial.transform import Rotation as R

from unitree_legged_sdk.env import Robot_py, AliengoCommand, AliengoState, dog_stand

from live_plot_client import LivePlotClient


np.set_printoptions(precision=3, suppress=True, floatmode="fixed", linewidth=300)


def main():
    setproctitle("play_aliengo")
    
    live_plot = LivePlotClient(zmq_addr="tcp://192.168.1.103:5555")

    dog_kp = 80.0
    dog_kd = 2.0
    dog_kp_list = [dog_kp] * 12
    dog_kd_list = [dog_kd] * 12

    debug = True
    # breakpoint()
    robot_dog = Robot_py(control_freq=500, debug=debug, window_size=8, kp=dog_kp_list, kd=dog_kd_list)
    robot_dog.start_control()

    try:
        for i in itertools.count():
            dog_state = robot_dog.get_state()
            rpy = R.from_euler("xyz", dog_state.rpy)
            gravity = rpy.inv().apply(np.array([0., 0., -1.]))
            
            live_plot.send(gravity.tolist())
            print(gravity)
            
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("End")
    except Exception as outer_e:
        # 捕获最外层的异常并进入调试
        import traceback

        traceback.print_exc()
        print(f"An unexpected error occurred: {outer_e}")
        breakpoint()  # 在最外层捕获所有异常并进入调试


if __name__ == "__main__":
    main()
