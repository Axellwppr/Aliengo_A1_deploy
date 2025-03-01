from a1_interface import A1ArmInterface
from play import Arm
from torchrl.envs.utils import set_exploration_type, ExplorationType
from typing import Union, Tuple
import argparse
import datetime
import itertools
import math
import numpy as np
import rospy
import time
import torch


class Arm_test(Arm):
    def step(self, joint_pos=None, joint_vel=None, tau_ff=None):
        joint_pos.clip_(-torch.pi, torch.pi)
        joint_pos.sub_(self.pos).clip(-0.2, 0.2).add_(self.pos)
        # print(joint_pos)
        self._arm.set_targets(
            joint_pos.tolist(),
            # self.pos.tolist(),
            [0, 0, 0, 0, 0, 0],
        )
        if tau_ff is not None:
            self._arm.set_feed_forward_torques(tau_ff.tolist())


class IK:
    def __init__(self, robot):
        self.robot = robot

    def compute(
        self,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        target_force: Union[torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.robot.update_fk()
        ee_pos, ee_vel = self.robot.ee_pos, self.robot.ee_vel
        joint_pos, joint_vel = self.robot.pos, self.robot.vel
        J = self.robot.J

        ee_pos_diff = target_pos - ee_pos
        ee_vel_diff = target_vel - ee_vel

        desired_vel = 40 * ee_pos_diff + 8 * ee_vel_diff
        nomial_error = -joint_pos

        J_T = J.T
        lambda_matrix = torch.eye(J.shape[1]) * 0.1
        A = J_T @ J + lambda_matrix
        b = J_T @ desired_vel.unsqueeze(-1)
        delta_q = torch.linalg.lstsq(A, b).solution.squeeze(-1) + 0.1 * nomial_error
        if target_force is not None:
            tau_ff = torch.pinverse(J) @ target_force
        else:
            tau_ff = None
        return joint_pos + delta_q * self.robot.dt, delta_q, tau_ff


def lemniscate(t: float, c: float):
    sin_t = math.sin(t)
    cos_t = math.cos(t)
    sin2p1 = sin_t**2 + 1
    x = torch.tensor([c * sin_t, cos_t, sin_t * cos_t]) / sin2p1
    return x


from pathgen import PathGenerator

if __name__ == "__main__":
    rospy.init_node("a1_arm_interface", anonymous=True)
    path = "/home/axell/桌面/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"
    torch.set_grad_enabled(False)

    arm = A1ArmInterface(kp=[120, 120, 80, 30, 30, 30], kd=[2, 2, 2, 1, 1, 0.4])
    dt = 0.02
    robot = Arm_test(arm=arm, dt=dt, urdf_path=path)
    ik = IK(robot)

    r = 0.15
    default_target_pos = torch.tensor([0.3, 0, 0.4])
    length = 2000
    for _ in range(10):
        pg = PathGenerator()
        his_pos = torch.zeros((length, 3))
        t = 0
        for i in itertools.count():
            if i % 100 == 0:
                print("rate = ", arm.count / (time.perf_counter() - arm.start_time))
            if i >= length:
                break
            t_start = time.perf_counter()
            if i % 1 == 0:
                t = i * dt
            # target_pos = default_target_pos  + torch.tensor([0.0, r *math.cos(t), r * math.sin(t)])
            # target_pos = default_target_pos  + lemniscate(t, 0.2) * torch.tensor([1.0, 0.4, 0.5])
            # target_vel = torch.tensor([0., r * -math.sin(t), r * math.cos(t)])
            target_pos = pg.get_position(t)
            # target_pos = default_target_pos
            # print(target_pos)
            target_vel = torch.tensor([0.0, 0.0, 0.0])
            joint_pos, joint_vel, tau_ff = ik.compute(target_pos, target_vel)
            his_pos[i] = target_pos
            # print(joint_pos)

            robot.step(joint_pos)
            step_time = time.perf_counter() - t_start
            # print("step_time", step_time)
            time.sleep(max(0, dt - step_time))

        # 获取当前时间并格式化
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        his_pos = his_pos.numpy()
        np.save(f"./his/his_pos_{formatted_time}.npy", his_pos)
