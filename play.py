import os
import sys
from typing import Optional
import time
import datetime
import numpy as np
import math
import itertools
from setproctitle import setproctitle

from env import CombinedActionManager, CombinedFlatEnv, JoyStickCommandManager, VRCommandManager
from arm.a1deploy.a1_new import Arm_py, ArmCommand, ArmState, arm_stand
from unitree_legged_sdk.env import Robot_py, AliengoCommand, AliengoState, dog_stand

from live_plot_client import LivePlotClient


np.set_printoptions(precision=3, suppress=True, floatmode="fixed", linewidth=300)

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
    
    live_plot = LivePlotClient(zmq_addr="tcp://192.168.1.100:5555")

    dog_kp = 80.0
    dog_kd = 2.0
    dog_kp_list = [dog_kp] * 12
    dog_kd_list = [dog_kd] * 12

    arm_kp_list = [50.0] * 3 + [20.0] * 3
    arm_kd_list = [2.0] * 3 + [1.0] * 3

    debug = False
    robot_arm = Arm_py(control_freq=500, debug=debug, window_size=20, kp=arm_kp_list, kd=arm_kd_list)
    robot_arm.start_control()
    
    robot_dog = Robot_py(control_freq=500, debug=debug, window_size=4, kp=dog_kp_list, kd=dog_kd_list)
    robot_dog.start_control()
    
    
    dog_default_joint_pos = np.array(
        [
            0.3, -0.3, 0.3, -0.3,
            1.0, 1.0, 1.1, 1.1,
            -2.0, -2.0, -2.1, -2.1,
        ],
    )
    # arm_default_joint_pos = np.array([0.0, 0.6, -0.6, 0.0, 0.0, 0.0])
    arm_default_joint_pos = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0])
    
    action_manager = CombinedActionManager(
        dog_robot=robot_dog,
        arm_robot=robot_arm,
        dog_default_jpos=dog_default_joint_pos,
        arm_default_jpos=arm_default_joint_pos,
    )
    command_manager = JoyStickCommandManager(
        dog_robot=robot_dog,
        arm_robot=robot_arm,
        urdf_path="/home/unitree/base_arm/arm/A1_SDK-arm/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"
    )
    # command_manager = VRCommandManager(
    #     dog_robot=robot_dog,
    #     arm_robot=robot_arm,
    #     urdf_path="/home/unitree/base_arm/arm/A1_SDK-arm/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
    #     ip="192.168.110.123"
    # )
    
    env = CombinedFlatEnv(
        dog_robot=robot_dog,
        arm_robot=robot_arm,
        action_manager=action_manager,
        command_manager=command_manager,
    )

    try:
        arm_stand(robot_arm, 5.0, arm_default_joint_pos)
        # arm_stand(robot_arm, 5.0, np.array([0.0, 0.6, -1.6, 0.0, 0.0, 0.0]))
        # breakpoint()
        dog_stand(
            robot=robot_dog,
            kp=dog_kp,
            kd=dog_kd,
            completion_time=5,
            default_joint_pos=dog_default_joint_pos,
        )
        print("Robot is now in standing position. Press Enter to exit...")

        input()
        input()

        print("start policy")

        # path = "/home/unitree/base_arm/ckp/policy-01-15_21-44-alienmforce-2040.onnx"
        # path = "/home/unitree/base_arm/ckp/policy-01-18_10-48-alienmforce-2083.onnx"
        path = "/home/unitree/base_arm/ckp/policy-01-18_19-50-alienmforce-2101.onnx" # continuous setpoint new clamp
        
        policy_module = ONNXModule(path)

        def policy(inp):
            out = policy_module(inp)
            action = out["action"].reshape(-1)
            carry = {k[1]: v for k, v in out.items() if k[0] == "next"}
            return action, carry

        obs_cmd, obs_plc = env.compute_obs()
        print(obs_cmd.shape, obs_plc.shape)

        policy_freq = 50
        dt = 1 / policy_freq

        inp = {
            "command_": obs_cmd[None, ...],
            "policy": obs_plc[None, ...],
            "is_init": np.array([True]),
            "adapt_hx": np.zeros((1, 128), dtype=np.float32),
            "context_adapt_hx": np.zeros((1, 128), dtype=np.float32),
        }
        
        import zmq
        context = zmq.Context()
        command_publisher = context.socket(zmq.PUB)
        command_publisher.bind("tcp://*:5557")
        
        for i in itertools.count():
            # print("Iter ", i)
            start = time.perf_counter()

            # if i % 4 == 0:
            obs_cmd, obs_plc = env.compute_obs()

            inp["command_"] = obs_cmd[None, ...]
            inp["policy"] = obs_plc[None, ...]
            inp["is_init"] = np.array([False], dtype=bool)

            action, carry = policy(inp)
            inp = carry

            env.apply_action(action)
            
            # live_plot.send({
            #     "jpos": robot_arm.jpos,
            #     "jpos_target": robot_arm.jpos_des
            # })
            arm_state = robot_arm.get_state()
            command_publisher.send_pyobj(arm_state.jpos[:5])
            # print(arm_state.jpos[:5].tolist())
            # live_plot.send(np.stack([robot_arm.jpos, arm_state.jpos, robot_arm.jpos_des], axis=1).tolist())
            assert robot_arm.jvel.shape == arm_state.jvel.shape == arm_state.jvel_diff.shape, (f"{robot_arm.jvel.shape}, {arm_state.jvel.shape}, {arm_state.jvel_diff.shape}")
            # live_plot.send(np.stack([robot_arm.jvel, arm_state.jvel, arm_state.jvel_diff], axis=1).tolist())
            # jpos = robot_arm.get_state().jpos
            # live_plot.send(action[[8, 13, 14, 15, 16]].tolist() + jpos[:5])

            elapsed = time.perf_counter() - start                
            time.sleep(max(0, 0.02 - elapsed))

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
