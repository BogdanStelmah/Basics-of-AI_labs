import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


def print_hi():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print("x:\n{}".format(x))

    eye = np.eye(4)
    print("массив NumPy:\n{}".format(eye))

    x = np.linspace(-10, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, marker="x")
    plt.show()


if __name__ == '__main__':
    print_hi()
