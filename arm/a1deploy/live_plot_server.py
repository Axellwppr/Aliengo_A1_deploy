import zmq
import threading
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import sys


class LivePlotServer(QtCore.QObject):
    data_received = QtCore.pyqtSignal(
        list
    )  # Signal to communicate with the main thread

    def __init__(self):
        super().__init__()
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Plotting")
        self.win.resize(800, 600)  # Optional: Adjust window size
        self.plots = []
        self.curves = []
        self.data = []
        self.index = []  # 用于存储 x 轴的索引

        # Set up ZeroMQ and the thread to receive data
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind("tcp://*:5555")
        self.thread = threading.Thread(target=self.receive_data)
        self.thread.daemon = True
        self.thread.start()

        # Connect the data_received signal to the update_plots slot
        self.data_received.connect(self.update_plots)

        # Set up a timer to regularly update the plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)  # Update every 50 milliseconds

    def receive_data(self):
        """Receive data in a separate thread and emit it to the main thread."""
        while True:
            data = self.socket.recv_pyobj()  # Receive data from ZeroMQ
            if not data:
                continue
            # Normalize data to be a list of lists
            if not isinstance(data[0], list):
                data = [data]
            self.data_received.emit(
                data
            )  # Emit the data as a signal to the main thread

    def update_plots(self, data):
        """Update the plots in the main thread based on the received data."""
        m = len(data)  # Number of curves per plot
        n = len(data[0])  # Number of plots

        # Check that all inner lists have the same length
        if not all(len(inner_list) == n for inner_list in data):
            print("All inner lists must have the same length")
            return

        if n != len(self.plots):
            # Re-initialize plots
            self.win.clear()
            self.plots = []
            self.curves = [[] for _ in range(n)]
            self.data = [[[] for _ in range(m)] for _ in range(n)]
            self.index = [[] for _ in range(n)]  # 初始化索引列表
            for i in range(n):
                p = self.win.addPlot(row=i, col=0)
                p.showGrid(x=True, y=True)
                curve_list = []
                for j in range(m):
                    pen = pg.mkPen(width=2, color=pg.intColor(j))
                    c = p.plot(pen=pen)
                    curve_list.append(c)
                self.curves[i] = curve_list
                self.plots.append(p)
                hline = pg.InfiniteLine(
                    pos=0, angle=0, pen=pg.mkPen(color="r", width=1)
                )
                p.addItem(hline)

        # Update data
        for i in range(n):
            for j in range(m):
                self.data[i][j].append(data[j][i])
                if len(self.data[i][j]) > 500:
                    self.data[i][j] = self.data[i][j][-500:]
            self.index[i].append(len(self.index[i]))  # 更新索引
            if len(self.index[i]) > 500:
                self.index[i] = self.index[i][-500:]

    def update(self):
        """Update the curves with the latest data."""
        if not self.curves:
            return
        n = len(self.plots)
        m = len(self.curves[0])
        for i in range(n):
            for j in range(m):
                self.curves[i][j].setData(self.data[i][j])
            # Adjust Y-axis range
            all_data = []
            for d in self.data[i]:
                all_data.extend(d)
            if all_data:
                min_y = min(all_data)
                max_y = max(all_data)
                padding = (max_y - min_y) * 0.1 if max_y != min_y else 1
                self.plots[i].setYRange(min_y - padding, max_y + padding)

    def run(self):
        self.app.exec_()


if __name__ == "__main__":
    plotter = LivePlotServer()
    plotter.run()
