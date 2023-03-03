import numpy as np


class SimpleModel:
    def __init__(self, noise_mean: float, noise_covariance: float, dt: float = 0.001):
        self.x = np.array([[0], [0]])
        self.noise_mean = noise_mean
        self.noise_covariance = noise_covariance
        self.A = np.array([[5, 3], [0, 5]])
        self.b = np.array([[0], [1]])
        self.c = np.array([1, 0])
        self.d = 0
        self.dt = dt

    def send_command(self, u: float):
        dx = np.dot(self.A, self.x) + self.b * u
        self.integrate_step(dx)

        return self._get_measurements()

    def integrate_step(self, dx: np.array):
        self.x = self.x + dx * self.dt

    def _get_measurements(self):
        return self.x + (np.random.rand(2, 1) - self.noise_mean) / self.noise_covariance

