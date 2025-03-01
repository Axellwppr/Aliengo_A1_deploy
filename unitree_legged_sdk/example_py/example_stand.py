#!/usr/bin/python

import sys
import time
import math
import numpy as np

np.set_printoptions(precision=3, suppress=True)

import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_path, 'lib/python/arm64'))
import aliengo_py

def stand(robot: aliengo_py.Robot, kp=60.0, kd=2.0, completion_time=5.0, default_joint_pos=None):
    robot_command = aliengo_py.AliengoCommand()
    robot_command.kp = [kp] * 12
    robot_command.kd = [kd] * 12
    robot_command.jpos_des = [0.0] * 12
    robot_command.jvel_des = [0.0] * 12
    robot_command.tau_ff = [0.0] * 12

    first_run = True
    st = None
    init_q = np.zeros(12, dtype=np.float32)

    # Standing position (adjusted for Aliengo)
    # stand_q = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5])
    stand_q = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5])
    if default_joint_pos is not None:
        stand_q[:] = default_joint_pos
    # reset pos [-0.2, 1.16, -2.7, 0.2, 1.16, -2.7, -0.2, 1.16, -2.78, 0.2, 1.16, -2.7]

    while True:
        if first_run:
            robot_state = robot.get_state()
            init_q[:] = robot_state.jpos
            if np.isnan(init_q).any():
                print("Initial q has nan, skip this run.")
                continue
            first_run = False
            st = time.time()

        robot_state = robot.get_state()
        # print("jpos: ", np.array(robot_state.jpos))
        # print("proj", robot_state.projected_gravity)
        # print("quat", robot_state.quat)
        # print(robot_state.lxy)
        
        t = time.time()
        alpha = max(0.0, min(1.0, (t - st) / completion_time))
        target_q = alpha * stand_q + (1 - alpha) * init_q
        robot_command.jpos_des = target_q.tolist()
        # print("cmd: ", np.array(robot_command.jpos_des))
        robot.set_command(robot_command)

        if t - st >= completion_time:
            print("Standing complete.")
            break

        elapsed_time = time.time() - t
        sleep_time = max(1 / 50 - elapsed_time, 0)
        actual_freq = 1 / (sleep_time + elapsed_time)
        # print("Elapsed time: ", elapsed_time)
        # print("Actual freq: ", actual_freq)
        # print("Sleep time: ", sleep_time)
        time.sleep(sleep_time)

def main():
    control_freq = 5000
    robot = aliengo_py.Robot(control_freq)
    robot.start_control()

    print("Robot initialized. Press Enter to start standing...")
    input()

    # stand(robot, kp=0.0, kd=2.0, completion_time=50.0)
    stand(robot, kp=60.0, kd=2.0, completion_time=5.0)

    print("Robot is now in standing position. Press Enter to exit...")
    input()

if __name__ == "__main__":
    main()