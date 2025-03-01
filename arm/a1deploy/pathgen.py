import math
import random
import torch

class PathGenerator:
    def __init__(self, parameter_ranges=None):
        self._init_lissajous()
        default_pos_range = [[0.3, 0.4], [-0.1, 0.1], [0.3, 0.4]]
        self.default_pos = torch.tensor([random.uniform(*default_pos_range[0]), random.uniform(*default_pos_range[1]), random.uniform(*default_pos_range[2])])

        self.min = torch.tensor([0.0, -0.3, 0.0])
        self.max = torch.tensor([0.5, 0.3, 0.6])
        # self.path_types = ['lissajous', 'rectangle', 'triangle']
        # self.path_type = random.choice(self.path_types)
        # if self.path_type == 'lissajous':
        #     self._init_lissajous()
        # elif self.path_type == 'rectangle':
        #     self._init_rectangle()
        # elif self.path_type == 'triangle':
        #     self._init_triangle()
    
    def _init_lissajous(self):
        default_ranges = {
            'A': (0.1, 0.2),       # Amplitude in x
            'B': (0.1, 0.2),       # Amplitude in y
            'C': (0.1, 0.2),       # Amplitude in z
            'a': (1/3, 1/2),        # Frequency in x
            'b': (1/3, 1/2),        # Frequency in y
            'c': (1/3, 1/2),        # Frequency in z
            'delta': (0, 2*math.pi)  # Phase shift
        }
        self.A = random.uniform(*default_ranges['A'])
        self.B = random.uniform(*default_ranges['B'])
        self.C = random.uniform(*default_ranges['C'])
        self.a = random.uniform(*default_ranges['a'])
        self.b = random.uniform(*default_ranges['b'])
        self.c = random.uniform(*default_ranges['c'])
        self.delta1 = random.uniform(*default_ranges['delta'])
        self.delta2 = random.uniform(*default_ranges['delta'])
        self.path_function = self._lissajous_path
    
    def _lissajous_path(self, t):
        x = self.A * math.sin(self.a * t + self.delta1)
        y = self.B * math.sin(self.b * t + self.delta2)
        z = self.C * math.sin(self.c * t)
        pos = torch.tensor([x, y, z]) + self.default_pos
        pos = pos.clamp(self.min, self.max)
        return pos
    
    def get_position(self, t):
        return self.path_function(t)