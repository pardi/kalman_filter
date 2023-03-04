import numpy as np


class KalmanFilt:
    def __init__(self, x0: np.array, dt: float = 0.01):
        self.Q = np.array([[1, 1], [0, 1]])
        self.R = np.array([[1, 0], [0, 1]])
        self.F = np.array([[1, 0], [0, 1]])
        self.B = np.array([[0], [1]])
        self.H = np.array([[1.0, 0.0], [0.0, 1.0]])

        self.P = np.array([[1, 0], [0, 1]])

        # Noises
        self.w = 0
        self.v = 0

        self.x = x0
        self.u = 0

        self.dt = dt

    def prediction(self) -> np.array:
        self.x = self.F @ self.x + self.B * self.u

        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x

    def update(self, z: np.array) -> np.array:
        y = z - self.H @ self.x
        K = self.P @ self.H.T @ np.linalg.inv(self.R + self.H @ self.P @ self.H.T)

        self.x = self.x + K @ y
        self.P = (np.identity(2) - K @ self.H) @ self.P

        return self.x
