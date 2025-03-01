import threading
import sys
import termios
import tty
import torch
import time


class KeyboardCommandManager:
    def __init__(self, step_size=0.1, apply_scaling=True, device="cpu"):
        self.step_size = step_size
        self.device = device
        with torch.device(device):
            self.command = torch.zeros(13)
            self.default_pos = torch.tensor([0.3, 0.0, 0.4])
            self.pose = torch.tensor([1.0, 0.0, 0.0])
            self.command_setpoint_pos_ee_b = self.default_pos
            self.command_setpoint_pos_ee_b_max = torch.tensor([0.8, 0.4, 0.7])
            self.command_setpoint_pos_ee_b_min = torch.tensor([0.2, -0.4, 0.35])
            self.setpoint_diff_min = torch.tensor([-0.1, -0.1, -0.1])
            self.setpoint_diff_max = torch.tensor([0.1, 0.1, 0.1])
            self.default_kp = torch.tensor([45.0, 45.0, 45.0])
            self.command_kp = self.default_kp  # 默认值
            self.command_kd = 2 * torch.sqrt(self.command_kp)
            self.command_kp_range = (40, 60)

        self.compliant_ee = False
        self.apply_scaling = apply_scaling
        self.mass = 1.0

        self.running = True
        self.input_thread = threading.Thread(target=self._input_listener)
        self.input_thread.daemon = True
        self.input_thread.start()
        print("[KeyboardCommandManager]: 控制台输入监听已启动")

    def _input_listener(self):
        while self.running:
            ch = self._getch()
            if ch == "\x1b":  # 转义字符
                # 读取接下来的两个字符
                next1 = self._getch()
                next2 = self._getch()
                if next1 == "[":
                    if next2 == "A":
                        self._on_key("up")
                    elif next2 == "B":
                        self._on_key("down")
                    elif next2 == "C":
                        self._on_key("right")
                    elif next2 == "D":
                        self._on_key("left")
            else:
                self._on_key(ch)

    def _getch(self):
        """从标准输入获取单个字符。"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            ch = sys.stdin.read(1)
        except Exception as e:
            print("读取字符时出错: ", e)
            ch = ""
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def _on_key(self, key):
        key = key.lower()
        if key == "up":
            self.command_setpoint_pos_ee_b[0] += self.step_size
        elif key == "down":
            self.command_setpoint_pos_ee_b[0] -= self.step_size
        elif key == "right":
            self.command_setpoint_pos_ee_b[1] += self.step_size
        elif key == "left":
            self.command_setpoint_pos_ee_b[1] -= self.step_size
        elif key == "w":
            self.command_setpoint_pos_ee_b[2] += self.step_size
        elif key == "s":
            self.command_setpoint_pos_ee_b[2] -= self.step_size
        elif key == "k":
            self.command_kp += 10
            self.command_kd = 2 * torch.sqrt(self.command_kp)
        elif key == "l":
            self.command_kp -= 10
            self.command_kd = 2 * torch.sqrt(self.command_kp)
        elif key == "r":
            self.reset()
            print("command reset")
        elif key == "c":
            self.compliant_ee = not self.compliant_ee
            print(f"Compliant EE 设置为 {self.compliant_ee}")

        # 将值限制在合理范围内
        self.command_setpoint_pos_ee_b.clip_(
            self.command_setpoint_pos_ee_b_min, self.command_setpoint_pos_ee_b_max
        )

        # 确保 command_kp 在限制范围内
        self.command_kp.clip_(*self.command_kp_range)
        self.command_kd = torch.sqrt(self.command_kp) * 2

        if key is not None:
            print(
                "set ee_pos=",
                self.command_setpoint_pos_ee_b,
                "kp=",
                self.command_kp,
            )

    def reset(self):
        self.command_setpoint_pos_ee_b = self.default_pos
        self.command_kp = self.default_kp
        self.command_kd = torch.sqrt(self.command_kp) * 2

    # @torch.compile
    def update(self, ee_pos: torch.Tensor, ee_vel: torch.Tensor) -> torch.Tensor:
        # check if ee_pos out of range
        if (ee_pos < self.command_setpoint_pos_ee_b_min).any() or (
            ee_pos > self.command_setpoint_pos_ee_b_max
        ).any():
            pass

        # real_command = (self.command_setpoint_pos_ee_b - ee_pos).clamp(
        #     self.setpoint_diff_min, self.setpoint_diff_max
        # )
        real_command = self.command_setpoint_pos_ee_b - ee_pos
        # print("ee_pos out of range!", ee_pos)
        self.command[0:3] = real_command

        self.command[3:6] = self.command_kp
        self.command[6:9] = self.command_kd
        self.command[9] = self.mass
        self.command[10:13] = self.pose

        if self.compliant_ee:
            self.command[0:6] = 0
        if self.apply_scaling:
            self.command[3:6] *= self.command[0:3]
            self.command[6:9] *= -ee_vel
        return self.command

    def close(self):
        self.running = False
        self.input_thread.join()


class RandomSampleCommandManager:
    def __init__(self, apply_scaling=True):
        self.apply_scaling = apply_scaling

        # 硬编码的采样范围和默认值
        self.setpoint_x_range = (0.2, 0.5)
        self.setpoint_y_range = (-0.18, 0.18)
        self.setpoint_z_range = (0.3, 0.6)
        self.kp_range = (30.0, 100.0)
        self.damping_ratio_range = (0.5, 1.5)
        self.virtual_mass_range = (0.8, 1.2)
        self.compliant_ratio = -0.0
        self.default_mass_ee = 1.0

        # 初始化命令参数
        self.command = torch.zeros(10)
        self.command_setpoint_pos_ee_b = torch.tensor([0.2, 0.0, 0.5])
        self.command_kp = torch.tensor([60.0, 60.0, 60.0])
        self.command_kd = 2 * torch.sqrt(self.command_kp)
        self.compliant_ee = False
        self.damping_ratio = torch.tensor([1.0])
        self.mass = self.default_mass_ee

        self.resample_prob = 0.007

        # 随机生成初始命令
        self.sample_command()

    def sample_command(self):
        # 随机采样末端执行器位置
        self.command_setpoint_pos_ee_b[0].uniform_(*self.setpoint_x_range)
        self.command_setpoint_pos_ee_b[1].uniform_(*self.setpoint_y_range)
        self.command_setpoint_pos_ee_b[2].uniform_(*self.setpoint_z_range)
        # 采样 Kp
        self.command_kp.uniform_(*self.kp_range)

        # 决定是否启用顺应性
        self.compliant_ee = torch.rand(1).item() < self.compliant_ratio

        # 基于 Kp 和阻尼比采样 Kd
        self.damping_ratio.uniform_(*self.damping_ratio_range)
        self.command_kd = 2.0 * torch.sqrt(self.command_kp) * self.damping_ratio

        # 采样虚拟质量
        self.mass = (
            torch.empty(1).uniform_(*self.virtual_mass_range).item()
            * self.default_mass_ee
        )

    def update(self, ee_pos: torch.Tensor, ee_vel: torch.Tensor) -> torch.Tensor:
        # 随机重新采样命令
        # print(torch.rand(1).item())
        if torch.rand(1).item() < self.resample_prob:
            self.sample_command()
        # 更新命令向量
        self.command[0:3] = self.command_setpoint_pos_ee_b - ee_pos
        self.command[3:6] = self.command_kp
        self.command[6:9] = self.command_kd
        self.command[9] = self.mass

        if self.compliant_ee:
            self.command[0:6] = 0

        if self.apply_scaling:
            self.command[3:6] *= self.command[0:3]
            self.command[6:9] *= -ee_vel

        return self.command

    def reset(self):
        # 重置命令参数到默认值
        self.command_setpoint_pos_ee_b = torch.tensor([0.2, 0.0, 0.5])
        self.command_kp = torch.tensor([60.0, 60.0, 60.0])
        self.command_kd = 2 * torch.sqrt(self.command_kp)
        self.compliant_ee = False
        self.mass = self.default_mass_ee

    def __str__(self):
        return (
            f"Setpoint Position: {self.command_setpoint_pos_ee_b.tolist()}, "
            f"Kp: {self.command_kp.tolist()}, Kd: {self.command_kd.tolist()}, "
            f"Compliant EE: {self.compliant_ee}, Mass: {self.mass}"
        )


if __name__ == "__main__":
    CommandManager = RandomSampleCommandManager()
    while True:
        print(CommandManager.update(torch.zeros(3), torch.zeros(3)))
        time.sleep(0.02)
