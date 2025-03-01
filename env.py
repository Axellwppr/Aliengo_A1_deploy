from arm.a1deploy.a1_new import Arm_py, ArmCommand, ArmState
from unitree_legged_sdk.env import Robot_py, AliengoCommand, AliengoState

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List
import numpy as np
import threading
import sys
import termios
import tty

import zmq

import pytorch_kinematics as pk
import torch

from vr import VRPosition

joint_order_real = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    'arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5', 'arm_joint6', 'gripper1_axis', 'gripper2_axis'
]

joint_order_sim = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'arm_joint1', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5', 'arm_joint6', 'gripper1_axis', 'gripper2_axis']

sim_to_real_idx = [joint_order_sim.index(jo) for jo in joint_order_real]
real_to_sim_idx = [joint_order_real.index(jo) for jo in joint_order_sim]

def convert_sim_to_real(jpos_sim):
    return [jpos_sim[i] for i in sim_to_real_idx]

def convert_real_to_sim(jpos_real):
    return [jpos_real[i] for i in real_to_sim_idx]

assert convert_real_to_sim(joint_order_real) == joint_order_sim
assert convert_sim_to_real(joint_order_sim) == joint_order_real

class LowerCommand:
    jpos_des: List[float]
    jvel_des: List[float]
    tau_ff: List[float]
    def __init__(self, nums):
        self.jpos_des = [0.0] * nums
        self.jvel_des = [0.0] * nums
        self.tau_ff = [0.0] * nums

def mix(a: np.ndarray, b: np.ndarray, alpha: float):
    return a * (1 - alpha) + b * alpha

def wrap_to_pi(a: np.ndarray):
    # wrap to -pi to pi
    a = np.mod(a + np.pi, 2 * np.pi) - np.pi
    return a

class CommandManager:
    def __init__(self, dog_robot, arm_robot, urdf_path: str):
        self._dog_robot = dog_robot
        self._arm_robot = arm_robot

        # ------------------- 狗控制相关参数 -------------------
        # 默认狗模式下：pos_diff 和 yaw_diff
        self.setpoint_pos_diff_b = np.array([0.0, 0.0])  # x, y
        self.yaw_diff = 0.0

        # PID / 增益相关
        self.kp_dog_min = 5.0
        self.kp_dog_max = 20.0
        self.kp_dog_yaw = 1.0
        self.kd_dog_yaw = 2.0 * np.sqrt(self.kp_dog_yaw)
        self.kp_dog_xy = 10.0
        self.kd_dog_xy = 2.0 * np.sqrt(self.kp_dog_xy)
        self.virtual_mass = 5.0
        self.virtual_inertia = 1.0
        self.max_pos_x_b = 2.0
        self.max_pos_y_b = 2.0
        self.max_angvel = 0.7

        # 记录当前狗的 yaw，用于控制相对差值
        self._init_yaw = self._dog_robot.get_state().rpy[2]
        self._target_yaw = 0.0
        self._angvel = 0.0

        # ------------------- 机械臂控制相关参数 -------------------
        # 初始末端绝对位置 (base坐标系)
        self.arm_default_pos = np.array([0.2, 0.0, 0.5])
        self.command_setpoint_pos_ee_b = self.arm_default_pos.copy()
        self.command_setpoint_pos_ee_b_diff = np.zeros(3)

        self.kp_arm_min = 60.0
        self.kp_arm_max = 80.0
        self.kp_arm = 70.0
        self.kd_arm = 2 * np.sqrt(self.kp_arm)
        self.arm_pos_max = np.array([0.5, 0.2, 0.6])
        self.arm_pos_min = np.array([0.2, -0.2, 0.4])
        self.mass_arm = 1.0

        # 机械臂正向运动学
        self.chain = pk.build_serial_chain_from_urdf(
            open(urdf_path, "rb").read(), "arm_seg6"
        )
        self.chain = self.chain.to(dtype=torch.float32)

        # 18维命令（或你自定义的格式）
        self.command = np.zeros(18, dtype=np.float32)

    def _arm_fk(self):
        arm_state = self._arm_robot.get_state()
        joint_pos = torch.tensor(arm_state.jpos, dtype=torch.float32)
        ret = self.chain.forward_kinematics(joint_pos, end_only=True)
        ee_pos = ret.get_matrix()[0, :3, 3]
        
        dog_state = self._dog_robot.get_state()
        base_rpy = dog_state.rpy

        rotation = R.from_euler('xy', [base_rpy[0], base_rpy[1]])

        ee_pos_np = ee_pos.cpu().numpy() + np.array([0.08, 0., 0.07])
        ee_pos_rotated = rotation.apply(ee_pos_np)
        
        return ee_pos_rotated

    def update_command(self):
        self.command[:] = 0.0  # 先清零

        # 机械臂：pos_diff = (desired_pos - current_pos)
        ee_pos_now = self._arm_fk()
        self.command_setpoint_pos_ee_b_diff = (
            self.command_setpoint_pos_ee_b - ee_pos_now
        )
        
        # 狗：pos_diff
        self.command[0:2] = self.setpoint_pos_diff_b[:2]
        self.command[2] = self.yaw_diff
        # 臂：pos_diff
        self.command[3:6] = self.command_setpoint_pos_ee_b_diff

        # 狗 kp
        self.command[6:8] = self.kp_dog_xy * self.setpoint_pos_diff_b
        self.command[8] = self.kp_dog_yaw * self.yaw_diff
        # 臂 kp
        self.command[9:12] = self.kp_arm * self.command_setpoint_pos_ee_b_diff

        # 狗 kd
        self.command[12:13] = self.kd_dog_xy
        self.command[13:14] = self.kd_dog_yaw
        # 臂 kd
        self.command[14] = self.kd_arm

        # mass / inertia
        self.command[15] = self.virtual_mass
        self.command[16] = self.virtual_inertia
        self.command[17] = self.mass_arm
        
    def update_value(self):
        raise NotImplementedError

    def update(self):
        self.update_value()
        self.update_command()

class JoyStickCommandManager(CommandManager):
    def __init__(self, dog_robot, arm_robot, urdf_path: str):
        super().__init__(dog_robot, arm_robot, urdf_path)

        # ------------------- 模式/切换控制 -------------------
        # 默认先进入狗模式
        self._control_mode = "dog"  
        self._r1_prev = False      # 检测手柄R1按键沿
        self._pressed_debounce = False

    def _arm_fk(self):
        arm_state = self._arm_robot.get_state()
        joint_pos = torch.tensor(arm_state.jpos, dtype=torch.float32)
        ret = self.chain.forward_kinematics(joint_pos, end_only=True)
        ee_pos = ret.get_matrix()[0, :3, 3]
        return ee_pos.cpu().numpy()

    def _toggle_mode(self):
        if self._control_mode == "dog":
            # 切到机械臂模式前，先记录当前机械臂末端实际位置
            # 这样摇杆可以从当前末端位置继续移动
            self._update_arm_actual_pos_as_setpoint()
            self._control_mode = "arm"
            print("[CommandManager] 切换到机械臂模式")
        else:
            self._control_mode = "dog"
            print("[CommandManager] 切换到狗模式")

    def _update_arm_actual_pos_as_setpoint(self):
        """把机械臂当前末端实际位置，作为下一步的指令起始位置。"""
        self.command_setpoint_pos_ee_b = self._arm_fk()

    def update_value(self):
        robot_state = self._dog_robot.get_state()
        
        l1_pressed = robot_state.buttons.get("L1", False)
        if l1_pressed:
            self._dog_robot.damp()
            self._arm_robot.damp()

        # ----------- 1) 读取手柄按键状态：R1 切换模式 -----------
        r1_pressed = robot_state.buttons.get("R1", False)
        # 检测边沿：从 False -> True
        if r1_pressed and not self._r1_prev and not self._pressed_debounce:
            self._toggle_mode()
            self._pressed_debounce = True
        # 如果 R1 没有按下，允许下次触发
        if not r1_pressed:
            self._pressed_debounce = False
        self._r1_prev = r1_pressed

        # 摇杆值
        # 假设 lxy[0], lxy[1] 分别代表 左摇杆 x / y
        #     rxy[0], rxy[1] 分别代表 右摇杆 x / y
        lx, ly = robot_state.lxy
        rx, ry = robot_state.rxy

        # ----------- 2) 不同模式下的控制逻辑 -----------
        if self._control_mode == "dog":
            # 2.1) 狗模式: 控制机器人平动 pos_diff 及 yaw_diff
            #     - 这里示例只用左摇杆控制 x,y 差分
            #     - 用右摇杆的 x 控制 yaw
            #     - 用右摇杆的 y 可能做别的功能，这里简单忽略或你可自定义
            # 2.1.1) 平动
            alpha = 0.2
            self.setpoint_pos_diff_b[0] = mix(
                self.setpoint_pos_diff_b[0],
                ly * self.max_pos_x_b,  # 注意：ly 可能在 [-1,1] 之间
                alpha
            )
            self.setpoint_pos_diff_b[1] = mix(
                self.setpoint_pos_diff_b[1],
                -lx * self.max_pos_y_b,  # 注意：lx 可能在 [-1,1] 之间
                alpha
            )
            # 2.1.2) 旋转
            self._angvel = mix(self._angvel, -rx * self.max_angvel, alpha)
            self._target_yaw += self._angvel * 0.02
            self.yaw_diff = wrap_to_pi(
                self._target_yaw + self._init_yaw - robot_state.rpy[2]
            )
            # 如果你不想加 yaw_diff，可以直接置 0
            self.yaw_diff = 0
            
            # kp
            self.kp_dog = (self.kp_dog_max - self.kp_dog_min) * 0.5 * ry + (self.kp_dog_max + self.kp_dog_min) * 0.5
            self.kd_dog = 2 * np.sqrt(self.kp_dog)

        else:
            # 2.2) 机械臂模式: 控制机械臂末端绝对位置
            #     - 这里用左摇杆控制 x,y，右摇杆控制 z（或者都由一个摇杆来控制 + 按键切换也行）
            step_size = 0.005  # 你可以视需要调节
            # 假设 left_y -> x, left_x -> y, right_y -> z
            # （仅供示例，具体映射可自行定）
            self.command_setpoint_pos_ee_b[0] += ly * step_size
            self.command_setpoint_pos_ee_b[1] += -lx * step_size
            # self.command_setpoint_pos_ee_b[2] += ry * step_size

            # 限制在合理范围
            self.command_setpoint_pos_ee_b = np.clip(
                self.command_setpoint_pos_ee_b,
                self.arm_pos_min,
                self.arm_pos_max
            )
            
            self.command_kp_arm = (self.kp_arm_max - self.kp_arm_min) * 0.5 * ry + (self.kp_arm_max + self.kp_arm_min) * 0.5
            self.command_kd_arm = 2 * np.sqrt(self.command_kp_arm)

class VRCommandManager(CommandManager):
    def __init__(self, dog_robot, arm_robot, urdf_path: str, ip: str):
        super().__init__(dog_robot, arm_robot, urdf_path)
        
        self.vr = VRPosition(ip_address=ip)
    
    def update_value(self):
        arm_pos, dog_pos, damping = self.vr.get_vr()
        dog_state = self._dog_robot.get_state()
        if damping:
            self._dog_robot.damp()
            self._arm_robot.damp()
        
        dog_pos = np.array(dog_pos)
        arm_pos = arm_pos[:3, 3].reshape(-1)
        
        self.command_setpoint_pos_ee_b[:] = np.clip(self.arm_default_pos + arm_pos, self.arm_pos_min, self.arm_pos_max)
        
        alpha = 0.2
        
        self.setpoint_pos_diff_b[0] = mix(
            self.setpoint_pos_diff_b[0],
            dog_pos[1] * self.max_pos_x_b,  # 注意：ly 可能在 [-1,1] 之间
            alpha
        )
        self.setpoint_pos_diff_b[1] = mix(
            self.setpoint_pos_diff_b[1],
            -dog_pos[0] * self.max_pos_y_b,  # 注意：lx 可能在 [-1,1] 之间
            alpha
        )
        
        self.yaw_diff = wrap_to_pi(
            self._init_yaw - dog_state.rpy[2]
        )

class CombinedActionManager:
    def __init__(self, dog_robot, arm_robot, dog_default_jpos, arm_default_jpos, kp=[80.0]*12, kd=[2.0]*12):
        self._dog_robot = dog_robot
        self._arm_robot = arm_robot
        
        self.dog_action_dim = 12
        self.arm_action_dim = 6
        self.action_dim = self.dog_action_dim + self.arm_action_dim
        
        self.action = np.zeros(self.action_dim)

        # 当前目标关节角（内部做平滑用）
        self._dog_default_jpos = np.array(dog_default_jpos, dtype=np.float32)
        self._arm_default_jpos = np.array(arm_default_jpos, dtype=np.float32)

        # 混合系数及缩放，可按需调
        self._alpha = 0.9
        self._dog_action_scaling = [0.1] * 4 + [0.5] * 8
        self._arm_action_scaling = 1.0

        self._dog_command = LowerCommand(12)
        self._dog_command.jpos_des = self._dog_default_jpos.tolist()
        self._dog_command.jvel_des = [0.0] * 12
        self._dog_command.tau_ff = [0.0] * 12

        self._arm_command = LowerCommand(6)
        self._arm_command.jpos_des = self._arm_default_jpos.tolist()
        self._arm_command.jvel_des = [0.0] * 6
        self._arm_command.tau_ff = [0.0] * 6

    def step(self, action_sim: np.ndarray):
        # breakpoint()
        action_real = action_sim[sim_to_real_idx[:self.action_dim]]
        
        action_real = action_real.clip(-6, 6)
        
        self.action = (1-self._alpha) * self.action + self._alpha * action_real
        
        dog_act = self.action[: self.dog_action_dim]
        arm_act = self.action[self.dog_action_dim :]

        # 1) 狗
        dog_action_raw = self._dog_action_scaling * dog_act + self._dog_default_jpos
        self._dog_command.jpos_des = dog_action_raw.tolist()
        self._dog_robot.set_command(self._dog_command)

        # 2) 机械臂
        arm_action_raw = self._arm_action_scaling * arm_act + self._arm_default_jpos
        self._arm_command.jpos_des = arm_action_raw.tolist()
        self._arm_robot.set_command(self._arm_command)

        return action_sim

class CombinedFlatEnv:
    action_buf_steps = 3
    gyro_steps = 4
    gravity_steps = 1
    jpos_steps = 3
    jvel_steps = 3

    dim_arm_joint = 8
    dim_dog_joint = 12
    
    dim_dog_joint_act = 12
    dim_arm_joint_act = 5
    
    dim_command = 18
    dim_osc = 12

    def __init__(
        self,
        dog_robot,
        arm_robot,
        command_manager,
        action_manager,
    ) -> None:
        self._dog_robot = dog_robot
        self._arm_robot = arm_robot
        self.command_manager: JoyStickCommandManager = command_manager
        self.action_manager = action_manager

        # --------------- 狗相关 ---------------
        # 多步缓存
        self.gyro_multistep = np.zeros((self.gyro_steps, 3))
        self.gravity_multistep = np.zeros((self.gravity_steps, 3))

        self.dog_jpos_sim = np.zeros(self.dim_dog_joint)
        self.dog_jvel_sim = np.zeros(self.dim_dog_joint)
        self.gyro = np.zeros(3)
        self.gravity = np.zeros(3)

        # --------------- 机械臂相关 ---------------
        self.arm_jpos_sim = np.zeros(self.dim_arm_joint)
        self.arm_jvel_sim = np.zeros(self.dim_arm_joint)
        
        # --------------- 合并joint ---------------
        self.merged_jpos = np.zeros(self.dim_arm_joint + self.dim_dog_joint)
        self.merged_jvel = np.zeros(self.dim_arm_joint + self.dim_dog_joint)
        
        # 多步缓存
        self.jpos_multistep = np.zeros((self.jpos_steps, self.dim_arm_joint + self.dim_dog_joint))
        self.jvel_multistep = np.zeros((self.jvel_steps, self.dim_arm_joint + self.dim_dog_joint))

        # --------------- Oscillator + command ---------------
        self.oscillator = Oscillator(use_history=False)

        # --------------- Action buffer ---------------
        # 这里 action_dim = 18 (狗 12 + 臂 6)，由 CombinedActionManager 决定
        self.action_buf = np.zeros((self.dim_dog_joint_act + self.dim_arm_joint_act, self.action_buf_steps))

        # --------------- 记录/观测 ---------------
        self.step_count = 0

        # creare a zmq publisher
        self.context = zmq.Context()
        self.command_publisher = self.context.socket(zmq.PUB)
        self.command_publisher.bind("tcp://*:5556")
    
    def __del__(self):
        self.command_publisher.close()
        self.context.term()

    def update_command(self):
        self.command_manager.update()
        self.oscillator.update_phis(self.command_manager.command)

    def update_policy(self):
        # 1) 获取 狗 状态
        dog_state = self._dog_robot.get_state()
        self.dog_jpos_sim[:12] = dog_state.jpos
        self.dog_jvel_sim[:12] = dog_state.jvel
        self.gyro[:] = dog_state.gyro
        # 2) 获取 机械臂 状态
        arm_state = self._arm_robot.get_state()
        self.arm_jpos_sim[:5] = arm_state.jpos[:5]
        self.arm_jvel_sim[:5] = arm_state.jvel_diff[:5]
        # 3) 合并joint
        self.merged_jpos = np.concatenate([self.dog_jpos_sim, self.arm_jpos_sim])[real_to_sim_idx]
        self.merged_jvel = np.concatenate([self.dog_jvel_sim, self.arm_jvel_sim])[real_to_sim_idx]
        
        # 投影重力
        rpy = R.from_euler("xyz", dog_state.rpy)
        gravity = rpy.inv().apply(np.array([0., 0., -1.]))
        self.gravity = gravity

        # 更新多步缓存 (jpos, jvel, gyro, gravity)
        self.jpos_multistep = np.roll(self.jpos_multistep, shift=1, axis=0)
        self.jpos_multistep[0] = self.merged_jpos

        self.jvel_multistep = np.roll(self.jvel_multistep, shift=1, axis=0)
        self.jvel_multistep[0] = self.merged_jvel

        self.gyro_multistep = np.roll(self.gyro_multistep, shift=1, axis=0)
        self.gyro_multistep[0] = self.gyro

        self.gravity_multistep = np.roll(self.gravity_multistep, shift=1, axis=0)
        self.gravity_multistep[0] = self.gravity

    def compute_obs(self):
        self.update_policy()
        self.update_command()

        obs_command_list = [
            self.command_manager.command[:],
            self.oscillator.get_osc()
        ]

        # 准备拼 obs
        obs_policy_list = [
            self.gravity_multistep.reshape(-1),        # shape (3 * gravity_steps)
            self.jpos_multistep.reshape(-1),           # shape (18 * jpos_steps)
            self.jvel_multistep.reshape(-1),           # shape (18 * jvel_steps)
            self.action_buf[:, :self.action_buf_steps].reshape(-1),        # shape (action_dim * action_buf_steps)
        ]
        
        obs_command = np.concatenate(obs_command_list, dtype=np.float32)
        obs_policy = np.concatenate(obs_policy_list, dtype=np.float32)
        
        command_data = {
            "setpoint_base_pos_diff_b": self.command_manager.setpoint_pos_diff_b,
            "yaw_diff": self.command_manager.yaw_diff,
            "setpoint_pos_ee_b": self.command_manager.command_setpoint_pos_ee_b,
            "kp_dog_xy": self.command_manager.kp_dog_xy,
            "kp_dog_yaw": self.command_manager.kp_dog_yaw,
            "kp_arm": self.command_manager.kp_arm,
            "kd_dog_xy": self.command_manager.kd_dog_xy,
            "kd_dog_yaw": self.command_manager.kd_dog_yaw,
            "kd_arm": self.command_manager.kd_arm,
        }
        self.command_publisher.send_pyobj(command_data)
        return obs_command, obs_policy

    def apply_action(self, action_sim: np.ndarray = None):
        if action_sim is not None:
            action_raw = np.zeros(self.action_manager.action_dim)
            action_raw[:self.dim_arm_joint_act + self.dim_dog_joint_act] = action_sim
            self.action_manager.step(action_raw)
            self.action_buf[:, 1:] = self.action_buf[:, :-1]
            self.action_buf[:, 0] = action_sim
            

class Oscillator:
    def __init__(self, use_history: bool = False):
        self.use_history = use_history
        self.phi = np.zeros(4)
        self.phi[0] = np.pi
        self.phi[3] = np.pi
        self.phi_dot = np.zeros(4)
        self.phi_history = np.zeros((4, 4))

    def update_phis(self, command, dt=0.02):
        omega = np.pi * 4.0
        move = True #np.abs(command[:3]).sum() > 0.1
        if move:
            dphi = omega + self._trot(self.phi)
        else:
            dphi = self._stand(self.phi)
        self.phi_dot[:] = dphi
        self.phi = (self.phi + self.phi_dot * dt) % (2 * np.pi)
        self.phi_history = np.roll(self.phi_history, 1, axis=0)
        self.phi_history[0] = self.phi

    def _trot(self, phi: np.ndarray):
        dphi = np.zeros(4)
        dphi[0] = (phi[3] - phi[0])
        dphi[1] = (phi[2] - phi[1]) + ((phi[0] + np.pi - phi[1]) % (2 * np.pi))
        dphi[2] = (phi[1] - phi[2]) + ((phi[0] + np.pi - phi[2]) % (2 * np.pi))
        dphi[3] = (phi[0] - phi[3])
        return dphi

    def _stand(self, phi: np.ndarray, target=np.pi * 3 / 2):
        dphi = 2.0 * ((target - phi) % (2 * np.pi))
        return dphi

    def get_osc(self):
        if self.use_history:
            phi_sin = np.sin(self.phi_history)
            phi_cos = np.cos(self.phi_history)
        else:
            phi_sin = np.sin(self.phi)
            phi_cos = np.cos(self.phi)
        osc = np.concatenate([phi_sin, phi_cos, self.phi_dot], axis=-1)
        if self.use_history:
            osc = osc.reshape(-1)
        return osc