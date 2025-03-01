import zmq
import time
import numpy as np

context = zmq.Context()
command_pub = context.socket(zmq.PUB)
command_pub.bind("tcp://*:5556")

data = {
    "setpoint_pos_base_b": np.array([0.5, 0.0]),
    "setpoint_yaw_diff": np.array([0.0]),
    "setpoint_pos_ee_b": np.array([0.4, 0.0, 0.5]),
}

data = [0.0, 0.0, 0.0]

while True:
    data["time_stamp"] = time.perf_counter()
    command_pub.send_pyobj(data)
    # command_pub.send_pyobj(data)
    time.sleep(0.01)