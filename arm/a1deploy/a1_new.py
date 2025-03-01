import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control
from typing import List, Tuple
from dataclasses import dataclass
import threading
import time
import numpy as np
import math
# -------------------------------
# 一些工具函数，参考 AlienGo 的写法
# -------------------------------
def _moving_average(data_buf: np.ndarray) -> np.ndarray:
    """对数据进行简单的移动平均。"""
    return np.mean(data_buf, axis=0)


def mix(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """混合两个向量，用于平滑过渡等。"""
    return a * (1 - alpha) + b * alpha


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """将角度限制在 [-pi, pi) 范围内。"""
    # 对齐 AlienGo 示例中的 wrap_to_pi 写法
    a = np.mod(a + np.pi, 2 * np.pi) - np.pi
    return a


# -------------------------------
# ArmCommand / ArmState
# 与 AlienGo 的 AliengoCommand、AliengoState 相对应
# -------------------------------
class ArmCommand:
    def __init__(self, jpos_des: List[float] = [0.0] * 6, jvel_des: List[float] = [0.0] * 6, tau_ff: List[float] = [0.0] * 6):
        # 这里假设机械臂有 6 个关节
        self.jpos_des = jpos_des
        self.jvel_des = jvel_des
        self.tau_ff = tau_ff

@dataclass
class ArmState:
    jpos: np.ndarray = np.zeros(6)
    jvel: np.ndarray = np.zeros(6)
    jvel_diff: np.ndarray = np.zeros(6)
    jtorque: np.ndarray = np.zeros(6)
    time: float = 0


class freq_checker:
    def __init__(self, print_rate: int = 50):
        self.print_rate = print_rate
        self.count = 0
        self.start_time = None
    
    def start(self):
        self.count = 0
        self.start_time = time.time()
    
    def trigger(self):
        self.count += 1
        if self.count % self.print_rate == 0:
            print(f"freq: {self.count / (time.time() - self.start_time)}")

# -------------------------------
# 机械臂主要控制类，模仿 AlienGo 的结构
# -------------------------------
class Arm_py:
    def __init__(
        self,
        control_freq: int = 200,            # 控制频率（发布/刷新频率）
        window_size: int = 10,              # 平滑时的移动窗口大小
        update_rate_hz: float = 500.0,     # _update_loop 的刷新速率
        debug: bool = True,
        kp = [80.0, 80.0, 80.0, 30.0, 30.0, 30.0],
        kd = [2, 2, 2, 1, 1, 1],
    ):
        # ROS Publisher/Subscriber 初始化
        rospy.init_node("a1_arm_interface", anonymous=True)
        self._pub = rospy.Publisher("/arm_joint_command_host", arm_control, queue_size=10)
        self._sub = rospy.Subscriber("/joint_states_host", JointState, self._joint_state_callback)
        self._ros_rate = rospy.Rate(control_freq)

        self._debug = debug
        self._running = False
        self._damping = False
        self.wait_init = True
        self.kp = kp
        self.kd = kd

        # 移动平均滤波相关
        self.window_size = window_size
        self.update_period = 1.0 / update_rate_hz * window_size
        self._buf_index = 0

        # 用于保存当前平滑后的状态
        self._smooth_state = ArmState()
        
        # 用于保存 ring buffer
        self.jpos = np.zeros(6, dtype=np.float32)
        self.jpos_des = np.zeros(6, dtype=np.float32)
        self.jvel = np.zeros(6, dtype=np.float32)
        
        self._field_buffers = {
            "jpos": np.zeros((window_size, 6)),
            "jvel": np.zeros((window_size, 6)),
            "jvel_diff": np.zeros((window_size, 6)),
            "jtorque": np.zeros((window_size, 6)),
        }

        # 日志记录需要
        self.step_count = 0
        
        self.seq_count = 0

        # 线程相关
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._cmd_lock = threading.Lock()
        self._state_lock = threading.Lock()

        # 用于暂存命令
        self._latest_command = ArmCommand()
        
        # freq checker
        self.callback_freq = freq_checker(100)
        self.callback_freq.start()

    def start_control(self) -> None:
        """启动主控制循环，对应 AlienGo 里的 start_control。"""
        
        self._running = True
        self._thread.start()
        # 等待部分时间，给线程预热
        time.sleep(1.0)
        # 或者也可以一直等待 self._running 完成 init 的标志
        print("Arm_py: Start control loop.")

    def stop_control(self) -> None:
        """停止主控制循环，对应 AlienGo 里的 stop_control。"""
        self._running = False
        if self._thread.is_alive():
            self._thread.join()
        print("Arm_py: Stop control loop.")

    def set_command(self, command) -> None:
        """
        设置当前想要的命令，与 AlienGo 类似：
        - command.jpos_des
        - command.jvel_des
        - command.tau_ff
        """
        now_pos = np.array(self.get_state().jpos)
        command_pos = np.array(command.jpos_des)

        new_pos = (command_pos - now_pos).clip(-0.3, 0.3) + now_pos
        jpos_des = new_pos.tolist()
        jvel_des = command.jvel_des
        tau_ff = command.tau_ff

        with self._cmd_lock:
            self._latest_command = ArmCommand(jpos_des=jpos_des, jvel_des=jvel_des, tau_ff=tau_ff)

    def get_state(self) -> ArmState:
        """获取当前的平滑后状态。"""
        with self._state_lock:
            smoothed_state = self._compute_smoothed_state()
            smoothed_state.timestamp = rospy.get_time()
            return smoothed_state

    def _joint_state_callback(self, msg: JointState):
        """订阅 /joint_states_host 得到机械臂当前状态。"""
        with self._state_lock:
            # self.callback_freq.trigger()
            idx = self._buf_index
            self.jpos = np.asarray(msg.position[:6])
            self.jvel = np.asarray(msg.velocity[:6])
            self._field_buffers["jvel_diff"][idx, :6] = (np.asarray(msg.position[:6]) - self._field_buffers["jpos"][idx, :6]) / self.update_period
            self._field_buffers["jpos"][idx, :6] = msg.position[:6]
            self._field_buffers["jvel"][idx, :6] = msg.velocity[:6]
            self._field_buffers["jtorque"][idx, :6] = msg.effort[:6]
            
            self._buf_index = (idx + 1) % self.window_size
            
            if self.wait_init:
                self._latest_command.jpos_des = msg.position[:6]
                self.wait_init = False

    def _update_loop(self):
        """核心循环，类似 AlienGo 示例。"""
        while self._running:
            self._publish_command()
            self._ros_rate.sleep()

    def _compute_smoothed_state(self) -> ArmState:
        """对 ring buffer 中的数据做移动平均等简单滤波，得到平滑后的机械臂状态。"""
        bf = self._field_buffers
        # 移动平均
        jpos_smoothed = _moving_average(bf["jpos"])
        jvel_smoothed = _moving_average(bf["jvel"])
        jvel_smoothed_diff = _moving_average(bf["jvel_diff"])
        jtorque_smoothed = _moving_average(bf["jtorque"])

        new_state = ArmState(
            jpos=jpos_smoothed,
            jvel=jvel_smoothed,
            jvel_diff=jvel_smoothed_diff,
            jtorque=jtorque_smoothed,
        )
        return new_state

    def _publish_command(self):
        """根据 self._latest_command 构造 arm_control 并发布。"""
        # print(self._latest_command.jpos_des)
        if self._debug:
            # 如果是 debug 模式，就不真正发布命令
            return
        if self.wait_init:
            return
        if self._damping:
            with self._cmd_lock:
                cmd_msg = arm_control()
                self.seq_count += 1
                cmd_msg.header.seq = self.seq_count
                cmd_msg.header.stamp = rospy.Time.now()
                cmd_msg.header.frame_id = "world"
                # 设置期望值
                cmd_msg.p_des = [0.0] * 6
                cmd_msg.v_des = [0.0] * 6
                cmd_msg.t_ff = [0.0] * 6
                # 设置 KP、KD
                cmd_msg.kp = [0.0] * 6
                cmd_msg.kd = [10.0] * 6
        else:
            with self._cmd_lock:
                cmd_msg = arm_control()
                self.seq_count += 1
                cmd_msg.header.seq = self.seq_count
                cmd_msg.header.stamp = rospy.Time.now()
                cmd_msg.header.frame_id = "world"
                # 设置期望值
                cmd_msg.p_des = self._latest_command.jpos_des
                cmd_msg.v_des = self._latest_command.jvel_des
                cmd_msg.t_ff = self._latest_command.tau_ff
                self.jpos_des = self._latest_command.jpos_des
                # 设置 KP、KD
                cmd_msg.kp = self.kp
                cmd_msg.kd = self.kd

        self._pub.publish(cmd_msg)
    def damp(self):
        with self._cmd_lock:
            # self._damping = True
            self._debug = True


def arm_stand(arm: Arm_py, duration: float = 5.0, target_state: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    """机械臂初始化"""
    now_state = np.array(arm.get_state().jpos)
    print(now_state)
    target_state = np.array(target_state)
    
    jvel = np.zeros_like(now_state)
    jtorque = np.zeros_like(now_state)
    
    dt = 0.02
    steps = int(duration / dt)
    for i in range(steps):
        progress = i / steps
        
        new_state = now_state * (1 - progress) + target_state * progress
        arm.set_command(ArmCommand(jpos_des = new_state.tolist()))
        
        # print(arm.get_state().jpos)
        
        time.sleep(dt)
    # print("Resetting Arm")
    # state = np.array(arm.get_state().jpos)
    # error = np.abs(target_state - state).sum()
    # while error > 1.0:
    #     state = np.array(arm.get_state().jpos)
    #     _state = state + np.clip(target_state - state, -0.1, 0.1)
    #     arm.set_command(ArmCommand(jpos_des = _state.tolist()))
    #     error = np.abs(target_state - state).sum()
    #     time.sleep(0.1)
    #     print(error)
    print("standup done")

if __name__ == "__main__":
    arm = Arm_py(debug=False)
    arm.start_control()
    
    arm_stand(arm, 5.0, [0.0, 0.6, -0.6, 0.0, 0.0, 0.0])
    arm.stop_control()