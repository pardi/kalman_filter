from models import SimpleModel
import matplotlib.pyplot as plt
import random
import numpy as np

def main():
    my_sys = SimpleModel(noise_mean=0, noise_covariance=100000000)

    X = []
    DX = []

    for idx in range(300):
        u = 0.1 * np.sin(2 * np.pi * idx / 100.0)

        x = my_sys.send_command(u)
        X.append(x[0])
        DX.append(x[1])

    fig, ax = plt.subplots(2)

    ax[0].plot(X)
    ax[1].plot(DX)
    plt.show()


if __name__ == "__main__":
    main()