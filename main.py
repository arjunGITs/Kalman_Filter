'''
Author: Arjun Pradeep
Content: Implementation of Kalman Filter from scratch

This code has been inspired from the video lectures on Kalman Filter by Prof Michel van Biezen: https://www.youtube.com/watch?v=CaCcOwJPytQ&list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT&index=1
and the video lectures of coding Kalman Filter https://www.youtube.com/watch?v=X42HqGthOqs&list=PLvKAPIGzFEr8n7WRx8RptZmC1rXeTzYtA&index=1
'''


import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter


plt.figure()

DT = 0.01
NUM_STEPS = 1000
MEAS_UPDATE_AFTER_EVERY_STEPS = 20

NUMVARS = 2
X_o = np.zeros(NUMVARS)
Xs = []
COVs = []
meas_xs = []
meas_vs = []
real_xs = []
real_vs = []

# Scenario simulation (Reality)
z_x = 1.0  # m
z_v = 0.5  # m/s

# Initial Conditions
X_o[0] = z_x + 1
X_o[1] = z_v - 2

kf = KalmanFilter(X_o)


for step in range(NUM_STEPS):
    if step <= 200:
        u = 0.0
    elif 200 < step <= 400:
        u = 0.1
    elif 400 < step <= 600:
        u = 0.2
    elif 600 < step <= 800:
        u = 0.0
    elif 800 < step <= 1000:
        u = -0.4

    z_x = z_x + (DT * z_v) + (0.5 * u * (DT ** 2))
    z_v = z_v + (u * DT)

    meas_x = z_x + np.random.randn() * np.sqrt(kf.var_z_x)
    meas_v = z_v + np.random.randn() * np.sqrt(kf.var_z_v)

    kf.predict(u=u, dt=DT)
    if step != 0 and step % MEAS_UPDATE_AFTER_EVERY_STEPS == 0:
        z = np.array([[meas_x],
                      [meas_v]])
        kf.update(z)

    COVs.append(kf.cov)
    Xs.append(kf.mean)

    real_xs.append(z_x)
    real_vs.append(z_v)

    meas_xs.append(meas_x)
    meas_vs.append(meas_v)

plt.subplot(2, 1, 1)
plt.title('X- Position vs time')
plt.plot(real_xs, c='b', linewidth='3.0')
plt.plot(meas_xs, c='g', linewidth='0.5')
plt.plot([X[0] for X in Xs], c='r', linewidth='1.5')
plt.plot([X[0] - 2 * np.sqrt(cov[0, 0]) for X, cov in zip(Xs, COVs)], c='r', linestyle='dashed', linewidth='0.5')
plt.plot([X[0] + 2 * np.sqrt(cov[0, 0]) for X, cov in zip(Xs, COVs)], c='r', linestyle='dashed', linewidth='0.5')

plt.subplot(2, 1, 2)
plt.title('Velocity vs time')
plt.plot(real_vs, c='b', label='Real values', linewidth='3.0')
plt.plot(meas_vs, c='g', label='Measured values', linewidth='0.5')
plt.plot([X[1] for X in Xs], c='r', label='Estimated values', linewidth='1.5')
plt.plot([X[1] - 2 * np.sqrt(cov[1, 1]) for X, cov in zip(Xs, COVs)], c='r', linestyle='dashed', label='Variance indicators', linewidth='0.5')
plt.plot([X[1] + 2 * np.sqrt(cov[1, 1]) for X, cov in zip(Xs, COVs)], c='r', linestyle='dashed', linewidth='0.5')

plt.legend()
plt.show()