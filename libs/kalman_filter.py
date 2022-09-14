import numpy as np


class KalmanFilter:
    def __init__(self, measurement, dt=0.1):
        # Noise
        R_std = 0.35
        Q_std = 0.04
        self.R = np.eye(2) * R_std ** 2
        self.Q = np.eye(4) * Q_std ** 2

        # Initiliza State
        # 4D state with initial position and initial velocity assigned
        self.X = np.array([[measurement[0]], [measurement[1]], [0.0], [0.0]])

        # self.Bu = np.zeros((4,4))
        # self.U = np.array([[0.0], [0.0], [0.0], [0.0]])

        # Initilize Error covariance Matrix
        gamma = 10
        self.P = gamma * np.eye(4)

        # State Transition matrix
        self.A = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.B = np.eye(4)
        self.C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        # Obtain prior estimates and error covariance from initial values

    def predict(self):
        # prior state estimate
        self.X = self.A @ self.X
        # prior-error covariance matrix
        self.P = self.A @ self.P @ self.A.T + self.B @ self.Q @ self.B.T

    def update(self, measurement: tuple):
        # kalman gain
        self.G = (
            self.P @ self.C.T @ np.linalg.inv((self.C @ self.P @ self.C.T + self.R))
        )
        # new state estimate
        self.X = self.X + self.G @ (
            np.array(measurement).reshape(-1, 1) - self.C @ self.X
        )
        # new-error covariance matrix
        self.P = (np.eye(self.A.shape[0]) - self.G @ self.C) @ self.P
