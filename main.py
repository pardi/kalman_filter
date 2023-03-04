import matplotlib.pyplot as plt
import numpy as np

from kalmanfilter import KalmanFilt
from models import SimpleModel


def main():
    my_sys = SimpleModel(noise_mean=0, noise_covariance=100)

    X = []
    X_real = []
    # DX = []
    DX_real = []
    X_kf = []
    DX_kf = []
    U = []

    kf = KalmanFilt(x0=np.array([[0.0], [0.0]]))
    N = 100
    flag = True
    u = 0

    for idx in range(N):
        t = idx / float(N)
        # f = 10

        if flag:
            u = 10 * np.sin(2 * np.pi * t)
            if u < 0:
                flag = False

        z = my_sys.send_command(u)
        x_real = my_sys.x

        x_kf = kf.prediction()

        if idx % 3 == 0:
            x_kf = kf.update(z)

        X.append(z[0])
        X_real.append(x_real[0])
        X_kf.append(x_kf[0])
        # DX.append(x[1])
        DX_real.append(x_real[1])
        DX_kf.append(x_kf[1])
        U.append(u)

    fig, ax = plt.subplots(3)
    #
    ax[0].plot(X, "b")
    ax[0].plot(X_real, "r")
    ax[0].plot(X_kf, "-c")
    # ax[1].plot(DX, "b")
    ax[1].plot(DX_real, "r")
    ax[1].plot(DX_kf, "-c")

    # ax[2].plot([x - y for x, y in zip(X, X_kf)], "b")
    ax[2].plot([x - y for x, y in zip(DX_kf, DX_real)], "b")
    plt.show()


if __name__ == "__main__":
    main()
