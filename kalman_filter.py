import numpy as np

class KalmanFilter:
    def __init__(self, X_o) -> None:
        # Initial Conditions
        self.x_o = X_o[0]
        self.v_o = X_o[1]
        self.NUMVARS = len(X_o)
        self.X = np.array([[self.x_o],
                        [self.v_o]])

        # Process errors
        self.var_x = 0.1 ** 2
        self.var_v = 0.1 ** 2
        self.covar_xv = 0.0
        self.P = np.array([[self.var_x, self.covar_xv],
                             [self.covar_xv, self.var_v]])
        self.Q = np.eye(self.NUMVARS)*(0.1 ** 2)

        # Measurement errors
        self.var_z_x = 0.1 ** 2
        self.var_z_v = 0.1 ** 2
        self.covar_z_xv = 0.0
        self.R = np.array([[self.var_z_x, self.covar_z_xv],
                             [self.covar_z_xv, self.var_z_v]])

        # Initialization
        self.H = np.identity(self.NUMVARS)

    def predict(self, u, dt):
        # x = A x + B u
        # P = A P At + Q

        self.X_minus = self.X
        self.P_minus = self.P

        A = np.array([[1, dt],
                      [0, 1]])
        B = np.array([[0.5 * dt ** 2],
                      [dt]])

        X_new = A.dot(self.X_minus) + B.dot(u)
        P_new = (A.dot(self.P_minus).dot(A.T)) + self.Q

        self.X = X_new
        self.P = P_new

    def update(self, z):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        y = z - self.H.dot(self.X)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        X_new = self.X + K.dot(y)
        P_new = (np.eye(self.NUMVARS) - K.dot(self.H)).dot(self.P)

        self.X = X_new
        self.P = P_new

    @property
    def cov(self) -> np.array:
        return self.P

    @property
    def mean(self) -> np.array:
        return self.X

    @property
    def pos(self) -> float:
        return self.X[0]

    @property
    def vel(self) -> float:
        return self.X[1]