from dataclasses import dataclass

@dataclass
class SimConfig:
    L: float = 10.0
    N: int = 256
    DT: float = 1e-3
    g: float = 2000.0
    # gamma: float = 0.01

    # parametry muru i gombki
    R_wall: float = 3.5
    slope_width: float = 0.2
    V_max: float = 200.0
    R_sponge: float = 3.8
    sponge_power: float = 20.0

    # spoon parameters
    A_spoon: float = 500.0
    sigma_spoon: float = 0.3

    # visualization params
    draw_every: int = 3
    ite_steps: int = 2000
    epsilon: float = 1e-5