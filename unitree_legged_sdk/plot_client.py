import zmq
import time
import numpy as np


class LivePlotClient:
    def __init__(self, zmq_addr="tcp://localhost:5555", send_interval=0.01):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(zmq_addr)
        self.send_interval = send_interval

    def test(self):
        while True:
            # 生成随机数据
            data = np.random.rand(3).tolist()  # 假设发送长度为3的浮点数列表
            self.socket.send_pyobj(data)
            time.sleep(self.send_interval)  # 控制发送速度

    def send(self, data):
        self.socket.send_pyobj(data)


if __name__ == "__main__":
    sender = LivePlotClient(zmq_addr="tcp://192.168.123.55:5555")
    sender.test()