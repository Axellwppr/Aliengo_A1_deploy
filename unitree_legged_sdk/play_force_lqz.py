import os
import sys

from env import default_joint_pos, Robot_py, FlatEnvLQZ, JointPositionAction, JoyStickForce_xy_kp
from example_py.example_stand import stand

from typing import Optional

import time
import datetime
import numpy as np
import math
import itertools
import argparse

# from scipy.spatial.transform import Rotation as R
# from tensordict import TensorDict

from setproctitle import setproctitle

np.set_printoptions(precision=3, suppress=True, floatmode="fixed", linewidth=300)


# from torchrl.envs.utils import set_exploration_type, ExplorationType

import onnxruntime as ort
import json


class ONNXModule:

    def __init__(self, path: str):

        self.ort_session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
        with open(path.replace(".onnx", ".json"), "r") as f:
            self.meta = json.load(f)
        self.in_keys = [
            k if isinstance(k, str) else tuple(k) for k in self.meta["in_keys"]
        ]
        self.out_keys = [
            k if isinstance(k, str) else tuple(k) for k in self.meta["out_keys"]
        ]

    def __call__(self, input):
        args = {
            inp.name: input[key]
            for inp, key in zip(self.ort_session.get_inputs(), self.in_keys)
            if key in input
        }
        outputs = self.ort_session.run(None, args)
        outputs = {k: v for k, v in zip(self.out_keys, outputs)}
        return outputs

def main():
    setproctitle("play_aliengo")

    robot = Robot_py(control_freq=500, debug=False, window_size=8, kp=[60.0]*12, kd=[2.0]*12)
    robot.start_control()
    # breakpoint()
    # input()
    
    # create managers
    command_manager = JoyStickForce_xy_kp(robot)
    action_manager = JointPositionAction(
        robot, alpha=0.75, action_scaling=0.5
    )

    # create env
    env = FlatEnvLQZ(
        robot=robot,
        command_manager=command_manager,
        action_manager=action_manager,
    )
    print("Environment created")

    # path = "./policy/policy-alienforce-1892-osc-large_rand.onnx"
    # path = "./policy/policy-alienforce-1894-no_osc.onnx"
    # path = "./policy/policy-alienforce-1898-no_osc-nogradpen-flag.onnx"
    path = "/home/unitree/base_arm/unitree_legged_sdk/policy/policy-01-18_19-39.onnx"
    
    policy_module = ONNXModule(path)

    def policy(inp):
        out = policy_module(inp)
        action = out["action"].reshape(-1)
        carry = {k[1]: v for k, v in out.items() if k[0] == "next"}
        return action, carry

    stand(
        robot=env.robot,
        kp=action_manager.kp,
        kd=action_manager.kd,
        completion_time=5,
        default_joint_pos=default_joint_pos,
    )
    print("Robot is now in standing position. Press Enter to exit...")
    
    # for i in itertools.count():
    #     env.update()
    #     print(env.projected_gravity)
    #     time.sleep(0.1)
    
    # input()
    # input()
    input()

    obs = env.reset()
    print(obs.shape)
    print(policy)

    cmd_dim = env.cmd_dim
    obs_dim = env.obs_dim

    print("cmd: ", cmd_dim, ", obs: ", obs_dim)

    command, obs = (
        obs[:cmd_dim],
        obs[cmd_dim:],
    )

    policy_freq = 50
    dt = 1 / policy_freq

    try:
            inp = {
                "command_": command[None, ...],
                "policy": obs[None, ...],
                "is_init": np.array([True]),
                "adapt_hx": np.zeros((1, 128), dtype=np.float32),
                "context_adapt_hx": np.zeros((1, 128), dtype=np.float32),
            }
            for i in itertools.count():
                print("Iter ", i)
                start = time.perf_counter()

                obs_new = env.compute_obs()
                command, obs = obs_new[:cmd_dim], obs_new[cmd_dim:]

                inp["command_"] = command[None, ...]
                inp["policy"] = obs[None, ...]
                inp["is_init"] = np.array([False], dtype=bool)

                action, carry = policy(inp)
                inp = carry

                env.apply_action(action)

                elapsed = time.perf_counter() - start
                if i % 20 == 0:
                    # print("command:", command_manager.command)
                    print("freq:", 1 / max(elapsed, dt))
                
                time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        print("End")


if __name__ == "__main__":
    main()
