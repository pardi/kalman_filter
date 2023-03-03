import numpy as np


class KalmanFilt:
    def __init__(self, x0: np.array, dt: float = 0.001):
        self.Q = np.array([[1, 0], [0, 1]])
        self.R = np.array([[1, 0], [0, 1]])
        self.F = np.array([[1, 0], [0, 1]])
        self.B = np.array([[0], [1]])
        self.H = np.array([[1, 0], [0, 1]])

        self.P = np.array([[1, 0], [0, 1]])

        # Noises
        self.w = 0
        self.v = 0

        self.x = x0
        self.u = 0

        self.dt = dt

    def prediction(self):
        x_hat = np.dot(self.F, self.x) + np.dot(self.B, self.u)
        P = np.dot(np.dot(self.F, np.linalg.pinv(self.P)), self.F.transpose()) + self.Q

        return x_hat, P

    def update(self, z: np.array, x_hat: np.array, P: np.array):
        y = z - np.dot(self.H, x_hat)
        K = np.dot(np.dot(P, self.H), np.linalg.inv(self.R + np.dot(np.dot(self.H, np.linalg.pinv(self.P)), self.H.transpose())))

        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.identity(2) - np.dot(K, self.H)), self.P)
        return self.x
