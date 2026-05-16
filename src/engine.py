import numpy as np
from PySide6.QtCore import QThread, Signal
from numba import njit


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

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.spoon_active = False
        self.spoon_x = 0.0
        self.spoon_y = 0.0
        self.prev_x = 0.0
        self.prev_y = 0.0

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

    def run(self):
        # === CONFIG (will be done separatetly) ===
        L = 10
        N = 256
        DX = L / N
        DT = 1e-3
        g = 2000.0
        gamma = 0.02
        # spoon parameters
        A_spoon = 500.0
        sigma_spoon = 0.3

        x = np.linspace(-L/2, L/2, N)
        k = np.fft.fftfreq(N, d=DX) * 2 * np.pi

        X, Y = np.meshgrid(x, x)
        kx, ky = np.meshgrid(k, k)

        # mur gładki (tanh)
        R_wall = 3.5
        R = np.sqrt(X**2 + Y**2)
        slope_width = 0.2
        V_max = 200.0
        V = (V_max / 2.0) * (1.0 + np.tanh((R - R_wall) / slope_width))
        # V = np.where(R**2 > R_wall**2, 500.0 ,0)
        # V = 0.5 * (X**2 + Y**2)

        # gombka
        sponge_mask = np.ones((N,N))
        R_sponge = 3.8
        u = 20.0
        sponge_mask[R > R_sponge] = np.exp(-u * DT * (R[R > R_sponge] - R_sponge)**2)

        # filtr antyaliasingowy
        k_max = np.pi / DX
        k_cutoff = (2.0 / 3.0) * k_max
        anti_alias_filter = np.where(kx**2 + ky**2 < k_cutoff**2, 1.0, 0.0)

        # random matrix to start
        ksi = np.random.rand(N,N) + 1j * np.random.rand(N, N)
        ksi = ksi / np.sqrt(np.sum(np.abs(ksi)**2) * DX**2)

        print('CALCULATING MEAN-FIELD GROUND STATE ...\n[', end='', flush=True)
        # first iterations for ITE (imaginary time evolution) to get base state
        for i in range(1000):
            if (i+1) % 100 == 0:
                print('#', end='', flush=True)
            ksi = ITE_real_space(ksi, DT, V, g)
            ksi_k = np.fft.fft2(ksi)
            ksi_k = ITE_momentum_space(ksi_k, DT, kx, ky)
            ksi = np.fft.ifft2(ksi_k)

            ksi = ksi / np.sqrt(np.sum(np.abs(ksi)**2) * DX**2)
        print(']\nDONE!\nTIME DEPENDENT GPE SOLUTION:\nREAL TIME EVOLUTION IN PROGRESS ...')
    

        # RTE loop
        while self.is_running:
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

                V_spoon = A_spoon * np.exp(-dist_sq / (2 * sigma_spoon**2))
                V_total = V + V_spoon        
            else:
                V_total = V

            # no dissipation
            # ksi = RTE_real_space(ksi, DT, V_total, g)
            # ksi_k = np.fft.fft2(ksi)

            # # antyaliasing
            # ksi_k = ksi_k * anti_alias_filter

            # ksi_k = RTE_momentum_space(ksi_k, DT, kx, ky)
            # ksi = np.fft.ifft2(ksi_k)

            # # gombka
            # ksi = ksi * sponge_mask

            # with disspiation
            ksi = DGPE_real_space(ksi, DT, V_total, g, gamma)
            ksi_k = np.fft.fft2(ksi)

            # antyaliasing
            ksi_k = ksi_k * anti_alias_filter

            ksi_k = DGPE_momentum_space(ksi_k, DT, kx, ky, gamma)
            ksi = np.fft.ifft2(ksi_k)

            # aplikacja gombki
            ksi = ksi * sponge_mask

            # normalizacja
            ksi = ksi / np.sqrt(np.sum(np.abs(ksi)**2) * DX**2)

            # WAŻNE to wysyła sygał do GUI
            self.frame_ready.emit(ksi.copy())

    def stop(self):
        self.is_running = False