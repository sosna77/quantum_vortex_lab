import numpy as np
from PySide6.QtCore import QThread, Signal
from numba import njit


@njit(cache=True)
def ITE_real_space(ksi, DT, V, g):
    return ksi * np.exp(-DT * (V + g * np.abs(ksi)**2))

@njit(cache=True)
def ITE_momentum_space(ksi_k, DT, kx, ky):
    return ksi_k * np.exp(-DT * 0.5 * (kx**2 + ky**2))

@njit(cache=True)
def RTE_real_space(ksi, DT, V, g):
    return ksi * np.exp(-1j * DT * (V + g * np.abs(ksi)**2))

@njit(cache=True)
def RTE_momentum_space(ksi_k, DT, kx, ky):
    return ksi_k * np.exp(-1j * DT * 0.5 * (kx**2 + ky**2))

class GPEWorker(QThread):

    frame_ready = Signal(object)

    def __init__(self):
        super().__init__()
        self.is_running = True

    def run(self):
        # === CONFIG (will be done separatetly) ===
        L = 10
        N = 256
        DX = L / N
        DT = 1e-2
        g =20

        x = np.linspace(-L/2, L/2, N)
        k = np.fft.fftfreq(N, d=DX) * 2 * np.pi

        X, Y = np.meshgrid(x, x)
        kx, ky = np.meshgrid(k, k)

        V = 0.5 * (X**2 + Y**2)

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
            ksi = RTE_real_space(ksi, DT, V, g)
            ksi_k = np.fft.fft2(ksi)
            ksi_k = RTE_momentum_space(ksi_k, DT, kx, ky)
            ksi = np.fft.ifft2(ksi_k)

            # WAŻNE to wysyła sygał do GUI
            self.frame_ready.emit(ksi.copy())

    def stop(self):
        self.is_running = False