import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit(cache=True)
def evolve_real_space(ksi, DT, V, g):
    return ksi * np.exp(-DT * (V + g * np.abs(ksi)**2))

@njit(cache=True)
def evolve_momentum_space(ksi_k, DT, kx, ky):
    return ksi_k * np.exp(-DT * 0.5 * (kx**2 + ky**2))

def main():
    L = 10
    N = 256
    DX = L / N
    DT = 1e-2
    g = 90

    x = np.linspace(-L/2, L/2, N)
    k = np.fft.fftfreq(N, d=DX) * 2 * np.pi

    X, Y = np.meshgrid(x, x)
    kx, ky = np.meshgrid(k, k)

    V = 0.5 * (X**2 + Y**2)

    ksi = np.random.rand(N,N) + 1j * np.random.rand(N, N)
    ksi = ksi / np.sqrt(np.sum(np.abs(ksi)**2) * DX**2)

    for _ in range(1000):
        ksi = evolve_real_space(ksi, DT, V, g)
        ksi_k = np.fft.fft2(ksi)
        ksi_k = evolve_momentum_space(ksi_k, DT, kx, ky)
        ksi = np.fft.ifft2(ksi_k)

        ksi = ksi / np.sqrt(np.sum(np.abs(ksi)**2) * DX**2)

    plt.imshow(np.abs(ksi)**2, origin='lower', extent=[-L/2, L/2, -L/2, L/2])
    plt.show()

if __name__ == '__main__':
    main()