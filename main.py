from models import SimpleModel
import matplotlib.pyplot as plt
from kalmanfilter import KalmanFilt
import numpy as np


def main():
    my_sys = SimpleModel(noise_mean=0, noise_covariance=10000)

    X = []
    X_real = []
    DX = []
    DX_real = []
    X_kf = []
    DX_kf = []

    kf = KalmanFilt(x0=np.array([.0, .0]))

    for idx in range(300):
        u = 0.1 * np.sin(2 * np.pi * idx / 100.0)

        x = my_sys.send_command(u)
        x_real = my_sys.x

        x_hat, P = kf.prediction()

        x_kf = kf.update(x, x_hat, P)

        X.append(x[0])
        X_real.append(x_real[0])
        X_kf.append(x_hat[0])
        DX.append(x[1])
        DX_real.append(x_real[1])
        DX_kf.append(x_hat[1])

    fig, ax = plt.subplots(2)

    ax[0].plot(X)
    ax[0].plot(X_real)
    ax[0].plot(X_kf)
    ax[1].plot(DX)
    ax[1].plot(DX_real)
    ax[1].plot(DX_kf)
    plt.show()


if __name__ == "__main__":
    main()