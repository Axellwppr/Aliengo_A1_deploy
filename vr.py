import numpy as np
import threading
import time

from oculus_reader.reader import OculusReader


class VRPosition:
    def __init__(
        self,
        ip_address: str = None,
        right_controller: bool = True,
    ):
        self.oculus_reader = OculusReader(ip_address=ip_address)
        self.controller_id = "r" if right_controller else "l"
        self.reset_state()
        self.arm_enable_button = 'RG'
        self.reset_button = "A"
        self.base_setpoint_button = "rightJS"
        self.damping_button = "B"
        self.poses = {}

        self.pose_aggregated = np.eye(4)
        self.last_pose_aggregated = np.eye(4)


        theta = np.pi / 180 * 45
        self.delta_rotation = np.array([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0],
        ]).T @ np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

        # Start State Listening Thread #
        self.thread = threading.Thread(target=self._update_internal_state)
        self.thread.start()

        self.lock = threading.Lock()

    def reset_state(self):
        self._state = {
            # "poses": {},
            "buttons": {"A": False, "B": False, "X": False, "Y": False, "RThU": False, "LThU": False, "RJ": False, "LJ": False, "RG": False, "LG": False, "RTr": False, "LTr": False},
            # "controller_on": True,
            "start_pose_this_hold": None,
        }
        self.update_sensor = True

    def _update_internal_state(self, num_wait_sec=5, hz=100):

        while True:
            # Regulate Read Frequency #
            time.sleep(1 / hz)

            with self.lock:
                # Read Controller
                poses, buttons = self.oculus_reader.get_transformations_and_buttons()
                if poses == {}:
                    continue
                self.poses = poses

                # Determine Control Pipeline #
                start_record = buttons[self.arm_enable_button] and not self._state["buttons"][self.arm_enable_button]
                end_record = not buttons[self.arm_enable_button] and self._state["buttons"][self.arm_enable_button]
                self._state["buttons"] = buttons

                if buttons[self.reset_button]:
                    self.last_pose_aggregated = np.eye(4)
                    self.pose_aggregated = np.eye(4)

                if end_record:
                    self._state["start_pose_this_hold"] = None
                    self.last_pose_aggregated = self.pose_aggregated

                if not buttons[self.arm_enable_button]:
                    continue

                if start_record:
                    assert self._state["start_pose_this_hold"] is None
                    self._state["start_pose_this_hold"] = poses[self.controller_id]

                delta = np.linalg.inv(self._state["start_pose_this_hold"]) @ poses[self.controller_id]
                delta_T = self.delta_rotation @ delta[:3, 3]
                self.pose_aggregated = self.last_pose_aggregated.copy()
                self.pose_aggregated[:3, 3] += delta_T

    def get_vr(self):
        with self.lock:
            return self.pose_aggregated, self._state["buttons"].get(self.base_setpoint_button, (0.0, 0.0)), self._state["buttons"].get(self.damping_button, False)


if __name__ == "__main__":
    vr_position = VRPosition()
    while True:
        print(vr_position.pose_aggregated)
        time.sleep(0.5)