from a1_interface import A1ArmInterface
from collections import deque
from command_manager import KeyboardCommandManager, RandomSampleCommandManager
from live_plot_client import LivePlotClient
from setproctitle import setproctitle
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType
from typing import Tuple
import argparse
import gc
import itertools
import pytorch_kinematics as pk
import rospy
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

try:
    profile
except NameError:

    def profile(func):
        return func


class Arm:
    def __init__(
        self,
        dt=0.02,
        prev_steps=3,
        arm=None,
        command_manager=None,
        urdf_path="",
        debug=True,
        default_joint_pos=torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0, 0.0]),
        action_dim=5,
        use_vel=True,
    ):
        print("init Arm")
        self.chain = pk.build_serial_chain_from_urdf(
            open(urdf_path, "rb").read(), "arm_seg6"
        )
        self.chain = self.chain.to(dtype=torch.float32, device=device)
        self.dt = dt
        self.action_dim = action_dim
        self.use_vel = int(use_vel)
        self.step_count = 0
        self._arm = arm
        self.command_manager = command_manager
        self.debug = debug

        with torch.device(device):
            self.default_joint_pos = default_joint_pos.to(device)
            self.pos = torch.zeros(6)
            self.vel = torch.zeros(6)
            self.now_pos = torch.zeros(6)
            self.now_vel = torch.zeros(6)
            self.ee_pos = torch.zeros(3)
            self.ee_vel = torch.zeros(3)
            self.prev_actions = torch.zeros((self.action_dim, prev_steps))
            self.last_action = torch.zeros(self.action_dim)
            self.command = torch.zeros(10)
            self.obs = torch.zeros(
                5 + 5 * self.use_vel
            )  # + prev_steps * self.action_dim)

        self.plot = LivePlotClient(zmq_addr="tcp://127.0.0.1:5555")
        if not self.debug:
            self._arm.start()
            while self._arm.wait_init:
                print("waiting for arm to be ready")
                time.sleep(1)
            time.sleep(1)

    def close(self):
        self._arm.stop()
        self.command_manager.close()

    def reset(self):
        self.command_manager.reset()

    @profile
    def take_action(self, action: torch.Tensor):
        self.step_count += 1
        action.clip_(-2 * torch.pi, 2 * torch.pi)

        self.prev_actions[:, 1:] = self.prev_actions[:, :-1]
        self.prev_actions[:, 0] = action
        self.last_action.mul_(0.1).add_(action * 0.9)

        target = self.default_joint_pos.clone()
        target[: self.action_dim].add_(self.last_action * 1.0).clip_(
            -torch.pi, torch.pi
        )
        # print(target - self.pos)

        target.sub_(self.pos).clip_(-0.3, 0.3).add_(self.pos)
        # print(target)
        if not self.debug:
            self._arm.set_targets(
                target.cpu(),
                torch.zeros(6),
            )

    @profile
    def compute_obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.update_fk()
        self.command = self.command_manager.update(self.ee_pos, self.ee_vel)
        self.obs[:5] = self.pos[:5]
        # print(self.obs[:5])
        if self.use_vel:
            self.obs[5:10] = self.vel[:5]
        #     self.obs[10:] = self.prev_actions.flatten()
        # else:
        #     self.obs[5:] = self.prev_actions.flatten()
        # print("now_pos", self.pos)
        return self.command, self.obs

    @profile
    def update_fk(self):
        self.pos[:], self.vel[:] = self._arm.get_joint_states()
        # self.now_pos, self.now_vel = self._arm.get_joint_states()
        # self.pos.mul_(0.2).add_(self.now_pos * 0.8)
        # self.vel.mul_(0.2).add_(self.now_vel * 0.8)
        ret = self.chain.forward_kinematics(self.pos, end_only=True)
        self.J = self.chain.jacobian(self.pos)[0, :3]

        self.ee_pos = ret.get_matrix()[0, :3, 3]
        # print(self.ee_pos)
        self.ee_vel = (self.J @ self.vel.unsqueeze(1)).squeeze(1)

        # print("ee_pos", self.ee_pos)
        # print("ee_vel", self.ee_vel)


def main():
    rospy.init_node("a1_arm_interface", anonymous=True)
    setproctitle("play_a1")

    # path = "policy-10-27_22-39.pt"
    # path = "policy-10-29_18-44.pt" # GOOD!
    path = "policy-12-01_16-22.pt"

    policy = torch.load(path, weights_only=False).to(device)
    # po = TensorDict(policy)

    # print(policy)
    # breakpoint()
    # return
    policy.module[0].set_missing_tolerance(True)
    torch.set_grad_enabled(False)

    try:
        arm = A1ArmInterface(kp=[80, 80, 80, 30, 30, 30], kd=[2, 2, 2, 1, 1, 1])
        dt = 0.02
        robot = Arm(
            dt=dt,
            arm=arm,
            command_manager=KeyboardCommandManager(device=device),
            # command_manager=RandomSampleCommandManager(),
            urdf_path="/home/axell/桌面/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
            debug=False,
            action_dim=5,
            use_vel=True,
        )
        robot.reset()
        cmd, obs = robot.compute_obs()

        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            td = (
                TensorDict(
                    {
                        "policy": obs,
                        "is_init": torch.tensor(1, dtype=bool),
                        "adapt_hx": torch.zeros(128),
                        "command": cmd,
                    },
                    [],
                )
                .unsqueeze(0)
                .to(device)
            )

            elapsed_times = deque(maxlen=100)

            init_t = time.perf_counter()
            for i in itertools.count(1):
                start = time.perf_counter()

                cmd, obs = robot.compute_obs()
                # print(cmd)
                td["next", "policy"] = obs.unsqueeze(0)
                td["next", "command"] = cmd.unsqueeze(0)
                td["next", "is_init"] = torch.tensor([0], dtype=bool, device=device)
                td = td["next"]

                policy(td)
                action = td["action"][0]
                # print(action)
                robot.take_action(action[:5])

                robot.plot.send(td["ext_rec"][0, :3].tolist())

                # breakpoint(

                # print(td["ext_rec"][0])

                # gc.collect()

                end = time.perf_counter()
                elapsed = end - start
                elapsed_times.append(elapsed)

                # if i % 100 == 0:
                #     avg_elapsed = sum(elapsed_times) / len(elapsed_times)
                #     min_elapsed = min(elapsed_times)
                #     max_elapsed = max(elapsed_times)
                #     print(
                #         f"Iteration {i}: Last 100 loops - Avg: {avg_elapsed:.6f}s, Min: {min_elapsed:.6f}s, Max: {max_elapsed:.6f}s"
                #     )

                if i % 100 == 0:
                    print(
                        "now_pos", robot.ee_pos, "pos_diff", cmd[:3], "action", action
                    )
                    print("rate:", i / (time.perf_counter() - init_t))
                elapsed_total = time.perf_counter() - start
                data = obs[:3].tolist()
                # data.append(elapsed_total)
                # print(data)
                # robot.plot.send(obs[:6].tolist())

                # print("elapsed_total", elapsed_total)
                # robot.plot.send([elapsed_total])
                sleep_time = max(0, dt - elapsed_total)
                time.sleep(sleep_time)
                # print(action)

    except KeyboardInterrupt:
        robot.close()
        print("End")


if __name__ == "__main__":
    main()
