#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <boost/bind.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>

#include "unitree_legged_sdk/unitree_joystick.h"
#include "unitree_legged_sdk/unitree_legged_sdk.h"

namespace py = pybind11;

constexpr uint16_t TARGET_PORT = 8007;
constexpr uint16_t LOCAL_PORT = 8082;
constexpr char TARGET_IP[] = "192.168.123.10";  // target IP address

const int LOW_CMD_LENGTH = 610;
const int LOW_STATE_LENGTH = 771;
const int SMOOTHING = 6;

namespace UNITREE_LEGGED_SDK {

class AliengoState {
 public:
  std::array<float, 12> jpos{};
  std::array<float, 12> jvel{};
  std::array<float, 12> jvel_diff{};
  std::array<float, 12> jtorque{};
  std::array<float, 4> quat{};
  std::array<float, 3> rpy{};
  std::array<float, 3> gyro{};
  std::array<float, 3> projected_gravity{0, 0, -1};
  std::array<float, 2> lxy{};
  std::array<float, 2> rxy{};
  std::map<std::string, bool> buttons;  // Joystick button states
  double timestamp;
};

class AliengoCommand {
 public:
  std::array<float, 12> jpos_des{0.0};
  std::array<float, 12> jvel_des{0.0};
  std::array<float, 12> kp{0.0};
  std::array<float, 12> kd{2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                           2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  std::array<float, 12> tau_ff{0.0};
};

class AliengoInterface {
 public:
  AliengoInterface(int control_freq = 500, bool verbose = false)
      : udp(LOCAL_PORT,
            TARGET_IP,
            TARGET_PORT,
            LOW_CMD_LENGTH,
            LOW_STATE_LENGTH) {
    udp.InitCmdData(low_cmd);
    low_cmd.levelFlag = LOWLEVEL;

    dt = 1.0 / control_freq;
    start_time = std::chrono::steady_clock::now();
    verbose_ = verbose;
  }

  ~AliengoInterface() {
    low_cmd.levelFlag = HIGHLEVEL;
    udp.SetSend(low_cmd);
    udp.Send();

    std::cout << "Reset to HIGHLEVEL control mode." << std::endl;

    if (loop_control) {
      delete loop_control;
    }
    if (loop_recv) {
      delete loop_recv;
    }
    if (loop_send) {
      delete loop_send;
    }
  }

  void start_control() {
    std::cout << "Initializing environment..." << std::endl;
    // InitEnvironment();
    std::cout << "Environment initialized." << std::endl;

    loop_control = new LoopFunc(
        "control", dt, boost::bind(&AliengoInterface::control_loop, this));
    loop_recv = new LoopFunc("udp_recv", dt, 3,
                             boost::bind(&AliengoInterface::udp_recv, this));
    loop_send = new LoopFunc("udp_send", dt, 3,
                             boost::bind(&AliengoInterface::udp_send, this));
    loop_rate_count = new LoopFunc(
        "rate_count", 1.0, 1, boost::bind(&AliengoInterface::rate_count, this));
    loop_control->start();
    loop_recv->start();
    loop_send->start();
    loop_rate_count->start();
  }

  AliengoState get_robot_state() {
    std::lock_guard<std::mutex> lock(state_mutex);
    return robot_state;
  }

  void set_robot_command(const AliengoCommand& command) {
    std::lock_guard<std::mutex> lock(cmd_mutex);
    robot_command = command;
  }

  double get_timestemp() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(now -
                                                                     start_time)
        .count();
  }

 private:
  void udp_recv() { udp.Recv(); }
  void udp_send() { udp.Send(); }
  void control_loop() {
    udp.GetRecv(low_state);
    update_state();
    update_command();
    udp.SetSend(low_cmd);
    cnt++;
  }

  void rate_count() {
    if (verbose_) {
      std::cout << "control loop rate: " << cnt << std::endl;
    }
    cnt = 0;
  }

  void update_state() {
    std::lock_guard<std::mutex> lock(state_mutex);

    auto now = std::chrono::steady_clock::now();
    double now_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(now -
                                                                  start_time)
            .count();
    double dt = now_seconds - robot_state.timestamp;
    robot_state.timestamp = now_seconds;

    // IMU data
    robot_state.quat = {
        low_state.imu.quaternion[0], low_state.imu.quaternion[1],
        low_state.imu.quaternion[2], low_state.imu.quaternion[3]};
    robot_state.rpy = {low_state.imu.rpy[0], low_state.imu.rpy[1],
                       low_state.imu.rpy[2]};
    robot_state.gyro = {low_state.imu.gyroscope[0], low_state.imu.gyroscope[1],
                        low_state.imu.gyroscope[2]};

    float w = robot_state.quat[0], x = robot_state.quat[1],
          y = robot_state.quat[2], z = robot_state.quat[3];
    float x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
    robot_state.projected_gravity[0] = 2 * (x * z - w * y);
    robot_state.projected_gravity[1] = 2 * (y * z + w * x);
    robot_state.projected_gravity[2] = -(w2 - x2 - y2 + z2);

    // Joint states
    for (int i = 0; i < 12; ++i) {
      jvel_diff_buffer[(cnt % SMOOTHING) * 12 + i] =
          (low_state.motorState[i].q - robot_state.jpos[i]) / dt;
      float jvel_diff = 0.0;
      for (int j = 0; j < SMOOTHING; ++j) {
        jvel_diff += jvel_diff_buffer[j * 12 + i];
      }
      robot_state.jvel_diff[i] = jvel_diff / SMOOTHING;
      robot_state.jpos[i] = low_state.motorState[i].q;
      robot_state.jvel[i] = low_state.motorState[i].dq;
      robot_state.jtorque[i] = low_state.motorState[i].tauEst;
    }

    // Joystick data
    xRockerBtnDataStruct joystick_data;
    memcpy(&joystick_data, &low_state.wirelessRemote,
           sizeof(xRockerBtnDataStruct));
    robot_state.lxy = {joystick_data.lx, joystick_data.ly};
    robot_state.rxy = {joystick_data.rx, joystick_data.ry};

  robot_state.buttons = {
      {"R1", static_cast<bool>(joystick_data.btn.components.R1)},
      {"L1", static_cast<bool>(joystick_data.btn.components.L1)},
      {"start", static_cast<bool>(joystick_data.btn.components.start)},
      {"select", static_cast<bool>(joystick_data.btn.components.select)},
      {"R2", static_cast<bool>(joystick_data.btn.components.R2)},
      {"L2", static_cast<bool>(joystick_data.btn.components.L2)},
      {"F1", static_cast<bool>(joystick_data.btn.components.F1)},
      {"F2", static_cast<bool>(joystick_data.btn.components.F2)},
      {"A", static_cast<bool>(joystick_data.btn.components.A)},
      {"B", static_cast<bool>(joystick_data.btn.components.B)},
      {"X", static_cast<bool>(joystick_data.btn.components.X)},
      {"Y", static_cast<bool>(joystick_data.btn.components.Y)},
      {"up", static_cast<bool>(joystick_data.btn.components.up)},
      {"right", static_cast<bool>(joystick_data.btn.components.right)},
      {"down", static_cast<bool>(joystick_data.btn.components.down)},
      {"left", static_cast<bool>(joystick_data.btn.components.left)},
  };
  }

  void update_command() {
    std::lock_guard<std::mutex> lock(cmd_mutex);

    std::array<float, 12> tau_pd;
    for (int i = 0; i < 12; ++i) {
      tau_pd[i] = robot_command.kp[i] *
                      (robot_command.jpos_des[i] - robot_state.jpos[i]) +
                  robot_command.kd[i] *
                      (robot_command.jvel_des[i] - robot_state.jvel[i]) +
                  robot_command.tau_ff[i];
    }

    // if (cnt % 100 == 0) {
    //     std::cout << std::fixed << std::setprecision(2) << "tau_pd:";
    //     for (int i = 0; i < 12; ++i) {
    //         std::cout << " " << tau_pd[i];
    //     }
    //     std::cout << "\n";
    //     std::cout << "tau_est:";
    //     for (int i = 0; i < 12; ++i) {
    //         std::cout << " " << low_state.motorState[i].tauEst;
    //     }
    //     std::cout << "\n";
    //     std::cout << "jpos_des:";
    //     for (int i = 0; i < 12; ++i) {
    //         std::cout << " " << robot_command.jpos_des[i];
    //     }
    //     std::cout << "\n" << std::endl;
    // }

    for (int i = 0; i < 12; ++i) {
      low_cmd.motorCmd[i].q = robot_command.jpos_des[i];
      low_cmd.motorCmd[i].dq = robot_command.jvel_des[i];
      low_cmd.motorCmd[i].Kp = robot_command.kp[i];
      low_cmd.motorCmd[i].Kd = robot_command.kd[i];
      low_cmd.motorCmd[i].tau = robot_command.tau_ff[i];
    }

    // for (int i = 0; i < 12; ++i) {
    //     low_cmd.motorCmd[i].Kp = 0.0;
    //     low_cmd.motorCmd[i].Kd = 0.0;
    //     low_cmd.motorCmd[i].tau = tau_pd[i];
    // }
  }

  static uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
    uint32_t xbit = 0;
    uint32_t data = 0;
    uint32_t CRC32 = 0xFFFFFFFF;
    const uint32_t dwPolynomial = 0x04c11db7;

    for (uint32_t i = 0; i < len; i++) {
      xbit = 1 << 31;
      data = ptr[i];
      for (uint32_t bits = 0; bits < 32; bits++) {
        if (CRC32 & 0x80000000) {
          CRC32 = (CRC32 << 1) ^ dwPolynomial;
        } else {
          CRC32 <<= 1;
        }
        if (data & xbit) {
          CRC32 ^= dwPolynomial;
        }
        xbit >>= 1;
      }
    }
    return CRC32;
  }

  UDP udp;
  float dt;  // 0.001 ~ 0.01
  long long cnt = 0;

  std::array<float, (12 * SMOOTHING)> jvel_diff_buffer{};

  LowCmd low_cmd;
  LowState low_state;

  AliengoState robot_state;
  AliengoCommand robot_command;
  std::mutex state_mutex, cmd_mutex;

  std::chrono::steady_clock::time_point start_time;

  LoopFunc* loop_control = nullptr;
  LoopFunc* loop_recv = nullptr;
  LoopFunc* loop_send = nullptr;
  LoopFunc* loop_rate_count = nullptr;

  bool verbose_;
};

}  // namespace UNITREE_LEGGED_SDK

PYBIND11_MODULE(aliengo_py, m) {
  py::class_<UNITREE_LEGGED_SDK::AliengoInterface>(m, "Robot")
      .def(py::init<int, bool>(
        ))
          // py::arg("control_freq") = 500, py::arg("verbose") = false))
      .def("start_control",
           &UNITREE_LEGGED_SDK::AliengoInterface::start_control)
      .def("get_state", &UNITREE_LEGGED_SDK::AliengoInterface::get_robot_state)
      .def("get_timestamp",
           &UNITREE_LEGGED_SDK::AliengoInterface::get_timestemp)
      .def("set_command",
           &UNITREE_LEGGED_SDK::AliengoInterface::set_robot_command);

  py::class_<UNITREE_LEGGED_SDK::AliengoState>(m, "AliengoState")
      .def(py::init<>())
      .def_readwrite("jpos", &UNITREE_LEGGED_SDK::AliengoState::jpos)
      .def_readwrite("jvel", &UNITREE_LEGGED_SDK::AliengoState::jvel)
      .def_readwrite("jvel_diff", &UNITREE_LEGGED_SDK::AliengoState::jvel_diff)
      .def_readwrite("jtorque", &UNITREE_LEGGED_SDK::AliengoState::jtorque)
      .def_readwrite("quat", &UNITREE_LEGGED_SDK::AliengoState::quat)
      .def_readwrite("rpy", &UNITREE_LEGGED_SDK::AliengoState::rpy)
      .def_readwrite("gyro", &UNITREE_LEGGED_SDK::AliengoState::gyro)
      .def_readwrite("projected_gravity",
                     &UNITREE_LEGGED_SDK::AliengoState::projected_gravity)
      .def_readwrite("lxy", &UNITREE_LEGGED_SDK::AliengoState::lxy)
      .def_readwrite("rxy", &UNITREE_LEGGED_SDK::AliengoState::rxy)
      .def_readwrite("buttons", &UNITREE_LEGGED_SDK::AliengoState::buttons)  // Expose buttons
      .def_readwrite("timestamp", &UNITREE_LEGGED_SDK::AliengoState::timestamp);

  py::class_<UNITREE_LEGGED_SDK::AliengoCommand>(m, "AliengoCommand")
      .def(py::init<>())
      .def_readwrite("jpos_des", &UNITREE_LEGGED_SDK::AliengoCommand::jpos_des)
      .def_readwrite("jvel_des", &UNITREE_LEGGED_SDK::AliengoCommand::jvel_des)
      .def_readwrite("kp", &UNITREE_LEGGED_SDK::AliengoCommand::kp)
      .def_readwrite("kd", &UNITREE_LEGGED_SDK::AliengoCommand::kd)
      .def_readwrite("tau_ff", &UNITREE_LEGGED_SDK::AliengoCommand::tau_ff);
}
