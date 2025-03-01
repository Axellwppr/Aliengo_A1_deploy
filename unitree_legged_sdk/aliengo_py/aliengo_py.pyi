#  aliengo_py/aliengo_py.pyi
from typing import List, Dict

class AliengoState:
    jpos: List[float]
    jvel: List[float]
    jvel_diff: List[float]
    jtorque: List[float]
    quat: List[float]
    rpy: List[float]
    gyro: List[float]
    projected_gravity: List[float]
    lxy: List[float]
    rxy: List[float]
    buttons: Dict[str, bool]
    control_mode: int
    timestamp: float


class AliengoCommand:
    jpos_des: List[float]
    jvel_des: List[float]
    kp: List[float]
    kd: List[float]
    tau_ff: List[float]


class Robot:
    def __init__(self, control_freq: int, verbose: bool) -> None: ...
    def start_control(self) -> None: ...
    def get_state(self) -> AliengoState: ...
    def set_command(self, command: AliengoCommand) -> None: ...
