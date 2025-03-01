from play import Arm
from a1_interface import A1ArmInterface
from command_manager import KeyboardCommandManager
import rospy
import torch
import math
from time import sleep


import random


class WaveGenerator:
    def __init__(self, initial=0.0, min_amp=0.5, max_amp=1.5, interval=1.0):
        self.key_points = [initial]
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.interval = interval
        self.current_segment = 0
        self.last_target = initial

    def get(self, t: float) -> float:
        # Determine the current segment based on time
        segment = int(t // self.interval)

        # Generate new key points if needed
        while len(self.key_points) <= segment + 1:
            # Alternate the sign
            sign = -1 if len(self.key_points) % 2 else 1
            # Generate a new target within the specified range
            new_target = sign * random.uniform(self.min_amp, self.max_amp)
            self.key_points.append(new_target)

        # Get start and end points for interpolation
        start = self.key_points[segment]
        end = self.key_points[segment + 1]

        # Calculate the progress within the current interval
        progress = (t % self.interval) / self.interval

        # Linear interpolation between start and end
        return start + (end - start) * progress


def main():
    rospy.init_node("a1_arm_interface", anonymous=True)
    data = []
    n = 0  # Specify the joint index you want to control

    # Initialize the WaveGenerator
    wave_gen = WaveGenerator(initial=0.0, min_amp=0.2, max_amp=1.8, interval=1.0)

    try:
        arm = A1ArmInterface(kp=[80, 80, 80, 30, 30, 30], kd=[2, 2, 2, 1, 1, 1])
        dt = 0.02
        robot = Arm(
            dt=dt,
            arm=arm,
            command_manager=KeyboardCommandManager(),
            urdf_path="/home/axell/桌面/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
            debug=False,
            default_joint_pos=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        freq = 50
        rate = rospy.Rate(freq)
        steps = freq * 20  # Run for 20 seconds

        for step in range(steps):
            t = step / freq  # Current time in seconds

            # Get the wave value for the current time
            wave_value = wave_gen.get(t)

            # Initialize positions with default values
            positions = torch.tensor(
                [0, 0.5, -0.5, 0, 0, 0],
                dtype=torch.float32,
            )

            # Update the specified joint with the wave value
            positions[n] = wave_value

            # Set targets with scaling if needed
            arm.set_targets(positions * 0.5, torch.zeros(6, dtype=torch.float32))

            # Retrieve joint positions and velocities
            j_pos = arm.joint_pos_raw.clone()
            j_vel = arm.joint_vel_raw.clone()

            # Print current joint position
            print(j_pos)

            # Send data for plotting or logging
            robot.plot.send([j_pos[n].item(), arm.arm_control_msg.p_des[n]])
            # robot.plot.send([j_pos[n], wave_value])

            # Append data for saving
            data.append(
                [j_pos[n].item(), j_vel[n].item(), arm.arm_control_msg.p_des[n]]
            )

            # Sleep to maintain the loop rate
            rate.sleep()
    except KeyboardInterrupt:
        robot.close()
        print("End")

    # Save the collected data
    datat = torch.tensor(data)
    torch.save(datat, f"{n}.pt")


if __name__ == "__main__":
    main()
