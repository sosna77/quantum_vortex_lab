import numpy as np
from PySide6.QtCore import QThread, Signal
from numba import njit
from config import SimConfig

@njit(cache=True)
def ITE_real_space(ksi, DT, V, g):
    return ksi * np.exp(-DT * (V + g * np.abs(ksi)**2))

@njit(cache=True)
def ITE_momentum_space(ksi_k, DT, kx, ky):
    return ksi_k * np.exp(-DT * 0.5 * (kx**2 + ky**2))


# bez dysypacji
@njit(cache=True)
def RTE_real_space(ksi, DT, V, g):
    return ksi * np.exp(-1j * DT * (V + g * np.abs(ksi)**2))

@njit(cache=True)
def RTE_momentum_space(ksi_k, DT, kx, ky):
    return ksi_k * np.exp(-1j * DT * 0.5 * (kx**2 + ky**2))

# opcja z dysypacją
@njit(cache=True)
def DGPE_real_space(ksi, DT, V, g, gamma):
    return ksi * np.exp(-(1j + gamma) * DT * (V + g * np.abs(ksi)**2))

@njit(cache=True)
def DGPE_momentum_space(ksi_k, DT, kx, ky, gamma):
    return ksi_k * np.exp(-(1j + gamma) * DT * 0.5 * (kx**2 + ky**2))

class GPEWorker(QThread):
    frame_ready = Signal(object)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True
        self.spoon_active = False
        self.spoon_x = 0.0
        self.spoon_y = 0.0
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.current_gamma = 0.01

    def set_gamma(self, new_gamma):
        self.current_gamma = new_gamma

    def update_spoon(self, x, y, active):
        if active and not self.spoon_active:
            self.prev_x = x
            self.prev_y = y
        elif active:
            self.prev_x = self.spoon_x
            self.prev_y = self.spoon_y

        self.spoon_x = x
        self.spoon_y = y
        self.spoon_active = active
    
    def trigger_reset(self):
        self.reset_requested = True

    def run(self):
        cfg = self.config
        DX = cfg.L / cfg.N

        x = np.linspace(-cfg.L/2, cfg.L/2, cfg.N)
        k = np.fft.fftfreq(cfg.N, d=DX) * 2 * np.pi

        X, Y = np.meshgrid(x, x)
        kx, ky = np.meshgrid(k, k)

        # mur gładki (tanh)
        R = np.sqrt(X**2 + Y**2)
        V = (cfg.V_max / 2.0) * (1.0 + np.tanh((R - cfg.R_wall) / cfg.slope_width))

        # gombka
        sponge_mask = np.ones((cfg.N,cfg.N))
        sponge_mask[R > cfg.R_sponge] = np.exp(-cfg.sponge_power * cfg.DT * (R[R > cfg.R_sponge] - cfg.R_sponge)**2)

        # filtr antyaliasingowy
        k_max = np.pi / DX
        k_cutoff = (2.0 / 3.0) * k_max
        anti_alias_filter = np.where(kx**2 + ky**2 < k_cutoff**2, 1.0, 0.0)

        # pętla życia
        while self.is_running:
            self.reset_requested = False

            # ==== START LUB RESET ==== (init + ITE)
            ksi = np.random.rand(cfg.N,cfg.N) + 1j * np.random.rand(cfg.N, cfg.N)       # random matrix to start
            ksi = ksi / np.sqrt(np.sum(np.abs(ksi)**2) * DX**2)

            print('CALCULATING MEAN-FIELD GROUND STATE ...\n', end='', flush=True)
            # first iterations for ITE (imaginary time evolution) to get base state
            prev_ksi = np.zeros_like(ksi)

            for i in range(cfg.ite_steps):
                # if (i+1) % 100 == 0:
                #     print('#', end='', flush=True)
                if not self.is_running or self.reset_requested:
                    break

                ksi = ITE_real_space(ksi, cfg.DT, V, cfg.g)
                ksi_k = np.fft.fft2(ksi)
                ksi_k = ksi_k * anti_alias_filter
                ksi_k = ITE_momentum_space(ksi_k, cfg.DT, kx, ky)
                ksi = np.fft.ifft2(ksi_k)

                ksi = ksi / np.sqrt(np.sum(np.abs(ksi)**2) * DX**2)
                if i % cfg.draw_every == 0:
                    self.frame_ready.emit(ksi.copy())

                    diff = np.max(np.abs(np.abs(ksi) - np.abs(prev_ksi)))

                    if diff < cfg.epsilon:
                        print(f'ITE CONVERGENCE REACHED IN {i} STEPS')
                        break
                    prev_ksi = ksi.copy()
            print('TIME DEPENDENT GPE SOLUTION:\nREAL TIME EVOLUTION IN PROGRESS ...\n')

            # ==== RTE ====
            step = 0
            while self.is_running and not self.reset_requested:
                # liczymy potencjał łyżki jak aktywna
                if self.spoon_active:
                    dx = self.spoon_x - self.prev_x
                    dy = self.spoon_y - self.prev_y
                    L2 = dx**2 + dy**2

                    if L2 == 0:
                        dist_sq = (X - self.spoon_x)**2 + (Y - self.spoon_y)**2
                    else:
                        # parametr rzutu (jak daleko na odciunku łączącym starą i nową pozycję myszki jestem)
                        t = ((X - self.prev_x) * dx + (Y - self.prev_y) * dy) / L2
                        t = np.clip(t, 0.0, 1.0)

                        proj_x = self.prev_x + t * dx
                        proj_y = self.prev_y + t * dy

                        dist_sq = (X - proj_x)**2 + (Y - proj_y)**2

                    V_spoon = cfg.A_spoon * np.exp(-dist_sq / (2 * cfg.sigma_spoon**2))
                    V_total = V + V_spoon        
                else:
                    V_total = V

                # no dissipation
                # ksi = RTE_real_space(ksi, cfg.DT, V_total, g)
                # ksi_k = np.fft.fft2(ksi)
                # # antyaliasing
                # ksi_k = ksi_k * anti_alias_filter
                # ksi_k = RTE_momentum_space(ksi_k, cfg.DT, kx, ky)
                # ksi = np.fft.ifft2(ksi_k)
                # # gombka
                # ksi = ksi * sponge_mask

                # SSFT with disspiation 
                ksi = DGPE_real_space(ksi, cfg.DT, V_total, cfg.g, self.current_gamma)
                ksi_k = np.fft.fft2(ksi)
                ksi_k = ksi_k * anti_alias_filter           # antyaliasing
                ksi_k = DGPE_momentum_space(ksi_k, cfg.DT, kx, ky, self.current_gamma)
                ksi = np.fft.ifft2(ksi_k)

                # aplikacja gombki
                ksi = ksi * sponge_mask

                # normalizacja
                ksi = ksi / np.sqrt(np.sum(np.abs(ksi)**2) * DX**2)

                # WAŻNE to wysyła sygał do GUI
                step +=1
                if step % cfg.draw_every == 0:
                    self.frame_ready.emit(ksi.copy())

    def stop(self):
        self.is_running = False