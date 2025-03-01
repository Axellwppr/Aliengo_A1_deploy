import rospy
import torch
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control
from typing import List, Tuple
import threading
from live_plot_client import LivePlotClient
import time
import math


class A1ArmInterface:
    def __init__(
        self,
        control_frequency: int = 200,
        kp: List[float] = [40, 40, 40, 20, 20, 20],
        kd: List[float] = [40, 40, 40, 1, 1, 1],
        urdf_path: str = "",
    ):
        self.urdf_path = urdf_path

        self.pub = rospy.Publisher(
            "/arm_joint_command_host", arm_control, queue_size=10
        )
        # self.plot = LivePlotClient(ip="127.0.0.1", port=9999)
        self.rate = rospy.Rate(control_frequency)
        self.wait_init = True
        self.arm_control_msg = arm_control()
        self.arm_control_msg.header.seq = 0
        self.arm_control_msg.header.stamp = rospy.Time.now()
        self.arm_control_msg.header.frame_id = "world"
        self.arm_control_msg.kp = kp
        self.arm_control_msg.kd = kd
        self.arm_control_msg.p_des = None
        self.arm_control_msg.v_des = None
        self.arm_control_msg.t_ff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.running = False
        self.thread = None

        self.count = 0
        self.start_time = time.perf_counter()

        self.joint_efforts_raw = torch.zeros(6)
        self.joint_pos_raw = torch.zeros(6)
        self.joint_vel_raw = torch.zeros(6)

        self.joint_pos_buffer = torch.zeros(10, 6)

        self.joint_state_lock = threading.Lock()
        self.arm_control_msg_lock = threading.Lock()

        # Subscribe to joint states
        self.joint_state_sub = rospy.Subscriber(
            "/joint_states_host", JointState, self._joint_state_callback
        )

    def _joint_state_callback(self, msg: JointState):
        with self.joint_state_lock:
            self.count += 1
            # roll over the buffer
            # print(msg)

            # print(self.joint_pos_buffer)
            self.joint_pos_raw[:6] = torch.as_tensor(msg.position[:6])
            self.joint_pos_buffer[1:] = self.joint_pos_buffer[:-1].clone()
            # print(self.joint_pos_buffer)
            self.joint_pos_buffer[0, :6] = self.joint_pos_raw[:6]
            self.joint_efforts_raw[:6] = torch.as_tensor(msg.effort[:6])
            self.joint_vel_raw[:6] = torch.as_tensor(msg.velocity[:6])

            if self.wait_init:
                self.arm_control_msg.p_des = msg.position
                self.arm_control_msg.v_des = [0.0] * 6
                self.start_time = time.perf_counter()
                self.wait_init = False

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._control_loop)
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            self.wait_init = True
            if self.thread:
                self.thread.join()

    def _control_loop(self):
        while self.running:
            if self.wait_init:
                pass
            else:
                with self.arm_control_msg_lock:
                    if (
                        self.arm_control_msg.p_des == None
                        or self.arm_control_msg.v_des == None
                    ):
                        pass
                    else:
                        # print("p_des", self.arm_control_msg.p_des)
                        self.arm_control_msg.header.seq += 1
                        self.arm_control_msg.header.stamp = rospy.Time.now()
                        self.pub.publish(self.arm_control_msg)
            self.rate.sleep()

    def set_targets(self, positions: torch.Tensor, velocities: torch.Tensor):
        if self.wait_init:
            return
        if positions.size(0) != 6 or velocities.size(0) != 6:
            raise ValueError("Both positions and velocities must have 6 elements")
        with self.arm_control_msg_lock:
            self.arm_control_msg.p_des = positions.tolist()
            self.arm_control_msg.v_des = velocities.tolist()
        # print(self.arm_control_msg.p_des)

    def set_feed_forward_torques(self, torques: torch.Tensor):
        if torques.size(0) != 6:
            raise ValueError("Torques must have 6 elements")
        self.arm_control_msg.t_ff = torques.tolist()

    def get_joint_states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with self.joint_state_lock:
            joint_pos = self.joint_pos_buffer.mean(dim=0)
            joint_vel = (self.joint_pos_buffer[:5] - self.joint_pos_buffer[5:]).mean(
                dim=0
            ) / 0.01
        return joint_pos, joint_vel


import itertools

if __name__ == "__main__":
    # set print precision
    try:
        rospy.init_node("a1_arm_interface", anonymous=True)
        arm_interface = A1ArmInterface(
            # kp=[80, 80, 80, 30, 30, 30],
            kp=[80.0, 80.0, 80.0, 30.0, 30.0, 30.0],
            kd=[2, 2, 2, 1, 1, 1],
            # kd=[15.0, 15.0, 15.0, 1.0, 1.0, 1.0],
        )
        arm_interface.start()
        while arm_interface.wait_init:
            print("waiting for arm to be ready")
            time.sleep(1)
        arm_interface.set_targets(
            torch.zeros(6, dtype=torch.float32),
            torch.zeros(6, dtype=torch.float32),
        )
        freq = 50
        rate = rospy.Rate(freq)
        plot = LivePlotClient(zmq_addr="tcp://127.0.0.1:5555")
        time.sleep(2)

        for iter in itertools.count():
            t = iter / freq
            cmd_pos = torch.tensor(
                [
                    0.2 * math.sin(2 * math.pi * t),
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            cmd_vel = torch.zeros(6)
            arm_interface.set_targets(cmd_pos, cmd_vel)
            # pos, vel = arm_interface.get_joint_states()
            pos = arm_interface.joint_pos_raw
            vel = arm_interface.joint_vel_raw
            eff_calc = 80 * (cmd_pos - pos) + 15 * (0 - vel)
            # plot.send([vel[:3].tolist(), arm_interface.joint_vel_raw[:3].tolist()])
            plot.send(
                [eff_calc[:3].tolist(), arm_interface.joint_efforts_raw[:3].tolist()]
            )
            # now = time.perf_counter()
            # print(arm_interface.count / (now - arm_interface.start_time))
            rate.sleep()

        arm_interface.stop()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
