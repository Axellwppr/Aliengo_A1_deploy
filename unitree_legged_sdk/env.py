import aliengo_py
import numpy as np
import threading
import time
from typing import Optional
# import h5py
from scipy.spatial.transform import Rotation as R
from plot_client import LivePlotClient
from example_py.example_stand import stand

Robot = aliengo_py.Robot
AliengoCommand = aliengo_py.AliengoCommand
AliengoState = aliengo_py.AliengoState
dog_stand = stand

ORBIT_JOINT_ORDER = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

SDK_JOINT_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


# fmt: off
default_joint_pos = np.array(
    [
        0.1, -0.1, 0.1, -0.1,
        0.8, 0.8, 0.8, 0.8,
        -1.5, -1.5, -1.5, -1.5,
    ],
)

# sdk2orbit_ = [SDK_JOINT_ORDER.index(j) for j in ORBIT_JOINT_ORDER]
# fmt on


def normalize(v: np.ndarray):
    return v / np.linalg.norm(v)


def mix(a: np.ndarray, b: np.ndarray, alpha: float):
    return a * (1 - alpha) + b * alpha

def wrap_to_pi(a: np.ndarray):
    # wrap to -pi to pi
    a = np.mod(a + np.pi, 2 * np.pi) - np.pi
    return a

def orbit_to_sdk(joints: np.ndarray):
    return np.flip(joints.reshape(3, 2, 2), axis=2).transpose(1, 2, 0).reshape(-1)

def sdk_to_orbit(joints: np.ndarray):
    return np.flip(joints.reshape(2, 2, 3), axis=1).transpose(2, 0, 1).reshape(-1)

print(sdk_to_orbit(np.array(SDK_JOINT_ORDER)) == np.array(ORBIT_JOINT_ORDER))
print(orbit_to_sdk(np.array(ORBIT_JOINT_ORDER)) == np.array(SDK_JOINT_ORDER))
print(orbit_to_sdk(default_joint_pos).reshape(4, 3))


def _moving_average(data_buf: np.ndarray) -> np.ndarray:
    return np.mean(data_buf, axis=0)

class Robot_py:
    def __init__(
        self,
        control_freq: int = 500,
        window_size: int = 4,
        update_rate_hz: float = 200.0,
        debug: bool = True,
        kp = [80.0] * 12,
        kd = [2.0] * 12
    ) -> None:
        self._robot = aliengo_py.Robot(control_freq, False)
        self._running = False
        self._debug = debug
        self._damping = False

        self.kp = kp
        self.kd = kd

        # 多步缓存大小
        self.window_size = window_size
        self.update_period = 1.0 / update_rate_hz

        # 在这里维护一个“当前平滑后的状态”
        # 注意：AliengoState 各字段都是 list[float]，也可用 np.array 存储，然后再转回 list
        self._smooth_state = AliengoState()
        self._init_empty_state(self._smooth_state)

        # 为了平滑，需要为每个字段创建一个 ring-buffer
        # 例如 jpos, jvel, jvel_diff, jtorque, quat, rpy, gyro, projected_gravity
        # 每个字段都是形如 (window_size, dim) 的数组，用于存放最近 window_size 次的原始数据
        self._field_buffers = {
            "jpos": np.zeros((window_size, 12)),   # 例如 12 关节
            "jvel": np.zeros((window_size, 12)),
            "jvel_diff": np.zeros((window_size, 12)),
            "jtorque": np.zeros((window_size, 12)),
            "quat": np.zeros((window_size, 4)),    # 四元数
            "rpy": np.zeros((window_size, 3)),     # 3 维
            "gyro": np.zeros((window_size, 3)),
            "projected_gravity": np.zeros((window_size, 3)),
        }
        self._buf_index = 0  # ring buffer 下标

        # 如果需要做日志记录，可以在这里初始化 dataset
        self.step_count = 0

        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        

    def _init_empty_state(self, state: AliengoState):
        # breakpoint()
        state.jpos = [0.0] * 12
        state.jvel = [0.0] * 12
        state.jvel_diff = [0.0] * 12
        state.jtorque = [0.0] * 12
        state.quat = [0.0] * 4
        state.rpy = [0.0] * 3
        state.gyro = [0.0] * 3
        state.projected_gravity = [0.0] * 3
        state.lxy = [0.0, 0.0]
        state.rxy = [0.0, 0.0]
        state.buttons = {}
        state.timestamp = 0.0

    def start_control(self) -> None:
        self._robot.start_control()
        # self._running = True
        self._thread.start()
        time.sleep(0.5)
        while not self._running:
            print("wait init...")
            time.sleep(0.5)
        print("init finished!")

    def stop_control(self) -> None:
        self._running = False
        self._thread.join()

    def set_command(self, command) -> None:
        jpos_des = (orbit_to_sdk(np.array(command.jpos_des))).tolist()
        jvel_des = (orbit_to_sdk(np.array(command.jvel_des))).tolist()
        tau_ff = (orbit_to_sdk(np.array(command.tau_ff))).tolist()
        
        command_raw = AliengoCommand()
        command_raw.jpos_des = jpos_des
        command_raw.jvel_des = jvel_des
        command_raw.tau_ff = tau_ff
        command_raw.kp = self.kp
        command_raw.kd = self.kd

        # print(command_raw.jpos_des)

        if self._debug or self._damping:
            return
        self._robot.set_command(command_raw)

    def get_state(self) -> AliengoState:
        return self._smooth_state

    def _update_loop(self):
        """
        线程循环，高频获取底层状态 -> 存到 ring-buffer -> 平滑 -> 写到 self._smooth_state -> 日志
        """
        self._running = True
        while self._running:
            t0 = time.perf_counter()
            raw_state = self._robot.get_state()  # 底层原始数据

            # 1. 将最新数据存放进 ring-buffer
            self._push_raw_data(raw_state)

            # 2. 对 ring-buffer 做平滑，然后更新 self._smooth_state
            self._smooth_state = self._compute_smoothed_state(raw_state)

            # 等待
            elapsed = time.perf_counter() - t0
            sleep_time = self.update_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _push_raw_data(self, state: AliengoState):
        idx = self._buf_index
        bf = self._field_buffers

        bf["jpos"][idx] = state.jpos
        bf["jvel"][idx] = state.jvel
        bf["jvel_diff"][idx] = state.jvel_diff
        bf["jtorque"][idx] = state.jtorque
        bf["quat"][idx] = state.quat
        bf["rpy"][idx] = state.rpy
        bf["gyro"][idx] = state.gyro
        bf["projected_gravity"][idx] = state.projected_gravity
        
        # data["jpos_200"].append(sdk_to_orbit(bf["jpos"][idx]).tolist())
        # data["jvel_200"].append(sdk_to_orbit(bf["jvel"][idx]).tolist())
        
        # 环状索引
        self._buf_index = (idx + 1) % self.window_size

    def _compute_smoothed_state(self, raw_state: AliengoState) -> AliengoState:
        """
        从 ring-buffer 中做平滑，然后将结果写入到一个新的 AliengoState 里返回。
        """
        bf = self._field_buffers

        # 以 window_size 的数据做一个平均
        # 注意：这里仅仅是演示滑动平均，可替换成别的滤波器
        jpos_smoothed = sdk_to_orbit(_moving_average(bf["jpos"]))
        jvel_smoothed = sdk_to_orbit(_moving_average(bf["jvel"]))
        jvel_diff_smoothed = sdk_to_orbit(_moving_average(bf["jvel_diff"]))
        jtorque_smoothed = sdk_to_orbit(_moving_average(bf["jtorque"]))
        quat_smoothed = _moving_average(bf["quat"])
        rpy_smoothed = _moving_average(bf["rpy"])
        gyro_smoothed = _moving_average(bf["gyro"])
        proj_g_smoothed = _moving_average(bf["projected_gravity"])

        smooth_state = AliengoState()
        self._init_empty_state(smooth_state)

        # 把平滑值放回 state
        smooth_state.jpos = jpos_smoothed.tolist()
        smooth_state.jvel = jvel_smoothed.tolist()
        smooth_state.jvel_diff = jvel_diff_smoothed.tolist()
        smooth_state.jtorque = jtorque_smoothed.tolist()
        smooth_state.quat = quat_smoothed.tolist()
        smooth_state.rpy = rpy_smoothed.tolist()
        smooth_state.gyro = gyro_smoothed.tolist()
        smooth_state.projected_gravity = proj_g_smoothed.tolist()

        # lxy, rxy, control_mode, timestamp 等, 可直接用原始值或也进行平滑
        smooth_state.lxy = raw_state.lxy
        smooth_state.rxy = raw_state.rxy
        smooth_state.buttons = raw_state.buttons
        smooth_state.timestamp = raw_state.timestamp

        return smooth_state

    def damp(self):
        if self._debug:
            return
        self._damping = True
        command_damp = AliengoCommand()
        command_damp.jpos_des = [0.0] * 12
        command_damp.jvel_des = [0.0] * 12
        command_damp.tau_ff = [0.0] * 12
        command_damp.kp = [0.0] * 12
        command_damp.kd = [10.0] * 12
        self._robot.set_command(command_damp)

class CommandManager:

    command_dim: int

    def __init__(self, robot: Robot_py) -> None:
        self._robot = robot
        self.command = np.zeros(self.command_dim)

    def update(self):
        pass


class JoyStickFlat(CommandManager):

    command_dim = 4

    max_speed_x = 1.0
    max_speed_y = 0.5
    max_angvel = 1.0

    def update(self):
        robot_state = self._robot.get_state()
        self.command[0] = mix(self.command[0], robot_state.lxy[1] * self.max_speed_x, 0.2)
        self.command[1] = mix(
            self.command[1], -robot_state.lxy[0] * self.max_speed_y, 0.2
        )
        self.command[2] = mix(
            self.command[2], -robot_state.rxy[0] * self.max_angvel, 0.2
        )
        self.command[3] = 0.0

class FixedCommandForce(CommandManager):
    command_dim = 10

    setpoint_pos_b = np.array([1.0, 0.0, 0.0])
    yaw_diff = 0.0

    kp = 5.0
    kd = 3.0
    virtual_mass = 3.0

    def update(self):
        self.command[:2] = self.setpoint_pos_b[:2]
        self.command[2] = self.yaw_diff
        self.command[3:5] = self.kp * self.setpoint_pos_b[:2]
        # self.kd = np.sqrt(self.kp) * 2
        self.command[5:8] = self.kd
        self.command[8] = self.kp * self.yaw_diff
        self.command[9] = self.virtual_mass
        
class JoyStickForce_xy_yaw(FixedCommandForce):
    max_pos_x_b = 1.0
    max_pos_y_b = 1.0
    max_angvel = 0.7

    def __init__(self, robot: aliengo_py.Robot) -> None:
        
        super().__init__(robot)
        self._init_yaw = self._robot.get_state().rpy[2]
        self._target_yaw = 0.0
        self._angvel = 0.0
    
    def update(self):
        robot_state = self._robot.get_state()

        self._angvel = mix(self._angvel, -robot_state.rxy[0] * self.max_angvel, 0.2)
        self._target_yaw += self._angvel * 0.02

        self.yaw_diff = wrap_to_pi(self._target_yaw + self._init_yaw - robot_state.rpy[2])
        self.yaw_diff = 0
        self.setpoint_pos_b[0] = mix(self.setpoint_pos_b[0], robot_state.lxy[1] * self.max_pos_x_b, 0.2)
        self.setpoint_pos_b[1] = mix(self.setpoint_pos_b[1], -robot_state.lxy[0] * self.max_pos_y_b, 0.2)
        
        super().update()
    
class JoyStickForce_xy_kp(FixedCommandForce):
    max_pos_x_b = 1.0
    max_pos_y_b = 1.0
    max_kp = 10.0
    min_kp = 2.0

    def update(self):
        robot_state = self._robot.get_state()
        
        self.kp = (robot_state.rxy[1] + 1) / 2 * (self.max_kp - self.min_kp) + self.min_kp
        
        self.setpoint_pos_b[0] = mix(self.setpoint_pos_b[0], robot_state.lxy[1] * self.max_pos_x_b, 0.2)
        self.setpoint_pos_b[1] = mix(self.setpoint_pos_b[1], -robot_state.lxy[0] * self.max_pos_y_b, 0.2)
        
        super().update()


class ActionManager:

    action_dim: int

    def __init__(self, robot: Robot_py) -> None:
        self._robot = robot
        self.robot_cmd = AliengoCommand()
        self.action = np.zeros(self.action_dim)

    def update(self):
        pass

    def step(self, action: np.ndarray) -> None:
        raise NotImplementedError


class JointPositionAction(ActionManager):

    action_dim = 12

    kp: float = 60
    kd: float = 2

    def __init__(
        self,
        robot: Robot_py,
        clip_joint_targets: float = 1.6,
        alpha: float = 0.9,
        action_scaling: float = 0.5,
    ) -> None:
        super().__init__(robot)

        self.robot_cmd.jpos_des = default_joint_pos.tolist()
        self.robot_cmd.jvel_des = [0.0] * 12
        self.robot_cmd.kp = [self.kp] * 12
        self.robot_cmd.kd = [self.kd] * 12
        self.robot_cmd.tau_ff = [0.0] * 12

        self.clip_joint_targets = clip_joint_targets
        self.alpha = alpha
        self.action_scaling = action_scaling

    def step(self, action_sim: np.ndarray) -> np.ndarray:
        action_sim = action_sim.clip(-6, 6)
        self.action = mix(self.action, action_sim, self.alpha)
        self.jpos_target = (self.action * self.action_scaling) + default_joint_pos
        self.robot_cmd.jpos_des = self.jpos_target.tolist()
        # print(self.robot_cmd.jpos_des)
        self._robot.set_command(self.robot_cmd)
        return action_sim


class EnvBase:

    dt: float = 0.02

    def __init__(
        self,
        robot: Robot_py,
        command_manager: CommandManager,
        action_manager: ActionManager,
    ) -> None:
        self.robot = robot
        self.robot_state = AliengoState()

        self.command_manager = command_manager
        self.action_manager = action_manager

    def reset(self):
        self.update()
        self.command_manager.update()
        return self.compute_obs()

    def apply_action(self, action: np.ndarray):
        raise NotImplementedError
        action = self.action_manager.step(action)
        self.action_buf[:, 1:] = self.action_buf[:, :-1]
        self.action_buf[:, 0] = action

        self.update()
        self.command_manager.update()
        return self.compute_obs()

    def update(self):
        """Update the environment state buffers."""
        raise NotImplementedError

    def compute_obs(self):
        """Compute the observation from the environment state buffers."""
        raise NotImplementedError

class FlatEnv(EnvBase):

    smoothing_length: int = 5
    smoothing_ratio: float = 0.4

    action_buf_steps = 3

    def __init__(
        self,
        robot: Robot,
        command_manager: CommandManager,
        action_manager: ActionManager,
        log_file = None
    ) -> None:
        super().__init__(
            robot=robot,
            command_manager=command_manager,
            action_manager=action_manager,
        )
        self.log_file = log_file

        # obs
        self.jpos_sdk = np.zeros(12)
        self.jvel_sdk = np.zeros(12)

        self.jpos_sim = np.zeros(12)
        self.jvel_sim = np.zeros(12)

        self.action_buf = np.zeros((action_manager.action_dim, self.action_buf_steps))

        # self.rpy = np.zeros(3)
        # self.angvel_history = np.zeros((3, self.smoothing_length))
        # self.angvel = np.zeros(3)
        self.projected_gravity_history = np.zeros((3, 1))
        self.projected_gravity = np.zeros(3)

        self.obs_dim = self.compute_obs().shape[-1]

        self.step_count = 0
        if self.log_file is not None:
            default_len = 50 * 60
            self.log_file.attrs["cursor"] = 0
            self.log_file.create_dataset("observation", (default_len, self.obs_dim), maxshape=(None, self.obs_dim))
            self.log_file.create_dataset("action", (default_len, 12), maxshape=(None, 12))

            self.log_file.create_dataset("rpy", (default_len, 3), maxshape=(None, 3))
            self.log_file.create_dataset("jpos", (default_len, 12), maxshape=(None, 12))
            self.log_file.create_dataset("jvel", (default_len, 12), maxshape=(None, 12))
            self.log_file.create_dataset("jvel_diff", (default_len, 12), maxshape=(None, 12))
            self.log_file.create_dataset("jpos_des", (default_len, 12), maxshape=(None, 12))
            self.log_file.create_dataset("tau_est", (default_len, 12), maxshape=(None, 12))
            self.log_file.create_dataset("quat", (default_len, 4), maxshape=(None, 4))
            self.log_file.create_dataset("linvel", (default_len, 3), maxshape=(None, 3))
            self.log_file.create_dataset("angvel", (default_len, 3), maxshape=(None, 3))

    def _maybe_log(self):
        if self.log_file is None:
            return
        self.log_file["observation"][self.step_count] = self.obs
        self.log_file["action"][self.step_count] = self.action_buf[:, 0]
        # self.log_file["angvel"][self.step_count] = self.angvel
        # self.log_file["linvel"][self.step_count] = self._robot.get_velocity()
        self.log_file["rpy"][self.step_count] = np.array(self.robot_state.rpy)
        self.log_file["jpos"][self.step_count] = self.jpos_sim
        self.log_file["jvel"][self.step_count] = self.jvel_sim
        self.log_file["jvel_diff"][self.step_count] = np.array(self.robot_state.jvel_diff)
        self.log_file["jpos_des"][self.step_count] = self.action_manager.jpos_target
        # self.log_file["tau_est"][self.step_count] = self.tau_sim
        self.log_file.attrs["cursor"] = self.step_count

        if self.step_count == self.log_file["jpos"].len() - 1:
            new_len = self.step_count + 1 + 3000
            print(f"Extend log size to {new_len}.")
            for key, value in self.log_file.items():
                value.resize((new_len, value.shape[1]))
        self.step_count += 1

    def update(self):
        self.robot_state = self.robot.get_state()

        self.jpos_sdk[:] = self.robot_state.jpos
        self.jvel_sdk[:] = self.robot_state.jvel

        self.jpos_sim[:] = sdk_to_orbit(self.jpos_sdk)
        self.jvel_sim[:] = sdk_to_orbit(self.jvel_sdk)

        # self.prev_rpy = self.rpy
        # self.rpy[:] = self.robot_state.rpy

        # self.angvel_history[:] = np.roll(self.angvel_history, 1, axis=1)
        # self.angvel_history[:, 0] = self.robot_state.gyro
        # self.angvel = mix(
        #     self.angvel, self.angvel_history.mean(axis=1), self.smoothing_ratio
        # )

        rpy = R.from_euler("xyz", self.robot_state.rpy)
        gravity = rpy.inv().apply(np.array([0., 0., -1.]))
        # self.projected_gravity_history[:] = np.roll(
        #     self.projected_gravity_history, 1, axis=1
        # )
        # self.projected_gravity_history[:, 0] = gravity # self.robot_state.projected_gravity
        gravity = np.array([0., 0., -1.])
        self.projected_gravity = gravity #normalize(self.projected_gravity_history.mean(1))

        # TODO: add latency measurements
        # self.latency = (datetime.datetime.now() - self._robot.timestamp).total_seconds()
        # self.timestamp = time.perf_counter()

    def compute_obs(self):
        self.update()
        self.command_manager.update()

        obs = [
            self.command_manager.command,
            self.projected_gravity,
            self.jpos_sim,
            self.jvel_sim,
            # sdk_to_orbit(np.array(self.robot_state.jvel_diff)),
            self.action_buf[:, : self.action_buf_steps].reshape(-1),
        ]
        self.obs = np.concatenate(obs, dtype=np.float32)
        return self.obs
    
    def apply_action(self, action_sim: np.ndarray = None):
        if action_sim is not None:
            self.action_manager.step(action_sim)
            self.action_buf[:, 1:] = self.action_buf[:, :-1]
            self.action_buf[:, 0] = action_sim
        self._maybe_log()

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


class FlatEnvLQZ(EnvBase):
    action_buf_steps = 3
    gyro_steps = 4
    gravity_steps = 3
    jpos_steps = 3
    jvel_steps = 3
    def __init__(
        self,
        robot: Robot_py,
        command_manager: CommandManager,
        action_manager: ActionManager,
    ) -> None:
        super().__init__(
            robot=robot,
            command_manager=command_manager,
            action_manager=action_manager,
        )
        
        # self.plot = LivePlotClient(zmq_addr="tcp://192.168.123.55:5555")

        self.jpos_multistep = np.zeros((self.jpos_steps, 12))
        self.jvel_multistep = np.zeros((self.jvel_steps, 12))
        self.gyro_multistep = np.zeros((self.gyro_steps, 3))
        self.gravity_multistep = np.zeros((self.gravity_steps, 3))

        self.oscillator = Oscillator(use_history=False)
        self.command = np.zeros(10, dtype=np.float32)
        # self.command = np.zeros(22, dtype=np.float32)

        self.jpos_sim = np.zeros(12)
        self.jvel_sim = np.zeros(12)
        self.gyro = np.zeros(3)
        self.gravity = np.zeros(3)
        

        self.action_buf = np.zeros((action_manager.action_dim, self.action_buf_steps))
        
        self.projected_gravity_history = np.zeros((3, 1))
        self.projected_gravity = np.zeros(3)

        self.cmd_dim = self.command.shape[0]
        self.obs_dim = self.compute_obs().shape[-1]

        self.step_count = 0

    def update_command(self):
        self.command[:10] = self.command_manager.command[:]
        # self.command[10] = 1
        # self.command[11:23] = self.oscillator.get_osc()
        # self.command[10:22] = self.oscillator.get_osc()
        # self.plot.send(self.command[10:22].tolist())

    def update(self):
        self.robot_state = self.robot.get_state()

        self.jpos_sim[:] = self.robot_state.jpos
        self.jvel_sim[:] = self.robot_state.jvel
        self.gyro[:] = self.robot_state.gyro
        # self.gravity[:] = self.robot_state.projected_gravity

        rpy = R.from_euler("xyz", self.robot_state.rpy)
        gravity = rpy.inv().apply(np.array([0., 0., -1.]))
        self.projected_gravity = gravity


        # data["jpos"].append(self.jpos_sim.tolist())
        # data["jvel"].append(self.jvel_sim.tolist())
        # data["gravity"].append(self.gravity.tolist())
        
        # self.plot.send(self.gravity.tolist())
        # print(self.gravity)

        self.jpos_multistep = np.roll(self.jpos_multistep, shift=1, axis=0)
        self.jpos_multistep[0] = self.jpos_sim
        self.jvel_multistep = np.roll(self.jvel_multistep, shift=1, axis=0)
        self.jvel_multistep[0] = self.jvel_sim
        self.gyro_multistep = np.roll(self.gyro_multistep, shift=1, axis=0)
        self.gyro_multistep[0] = self.gyro
        self.gravity_multistep = np.roll(self.gravity_multistep, shift=1, axis=0)
        self.gravity_multistep[0] = self.projected_gravity


        self.command_manager.update()
        self.oscillator.update_phis(self.command_manager.command)
        self.update_command()

    def compute_obs(self):
        self.update()
        jpos_multistep = self.jpos_multistep.copy()
        # jpos_multistep[1:] = jpos_multistep[1:] - jpos_multistep[:-1]
        jvel_multistep = self.jvel_multistep.copy()
        # jvel_multistep[1:] = jvel_multistep[1:] - jvel_multistep[:-1]
        # print(self.jvel_multistep)
        obs = [
            self.command,
            # self.gyro_multistep.reshape(-1),
            self.gravity_multistep.reshape(-1),
            jpos_multistep.reshape(-1),
            jvel_multistep.reshape(-1),
            self.action_buf[:, :3].reshape(-1),
        ]
        self.obs = np.concatenate(obs, dtype=np.float32)
        return self.obs

    def apply_action(self, action_sim: np.ndarray = None):
        if action_sim is not None:
            self.action_manager.step(action_sim)
            self.action_buf[:, 1:] = self.action_buf[:, :-1]
            self.action_buf[:, 0] = action_sim
            # self.plot.send(action_sim.tolist())