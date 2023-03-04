import matplotlib.pyplot as plt
import numpy as np

from kalmanfilter import KalmanFilt
from models import SimpleModel


def main():
    my_sys = SimpleModel(noise_mean=0, noise_covariance=100)

    X = []
    X_real = []
    DX = []
    DX_real = []
    X_kf = []
    DX_kf = []

    kf = KalmanFilt(x0=np.array([[0.0], [0.0]]))

    for idx in range(100):
        u = 10 * np.sin(2 * np.pi * idx / 100.0)

        x = my_sys.send_command(u)
        x_real = my_sys.x

        x_hat = kf.prediction()

        x_kf = kf.update(x, x_hat)

        X.append(x[0])
        X_real.append(x_real[0])
        X_kf.append(x_kf[0])
        DX.append(x[1])
        DX_real.append(x_real[1])
        DX_kf.append(x_kf[1])

    fig, ax = plt.subplots(2)

    ax[0].plot(X, "b")
    ax[0].plot(X_real, "r")
    ax[0].plot(X_kf, "-cx")
    ax[1].plot(DX, "b")
    ax[1].plot(DX_real, "r")
    ax[1].plot(DX_kf, "-cx")
    plt.show()


if __name__ == "__main__":
    main()
