import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fuzzy_system import run_fuzzy_control_system

def plot_3d():
    x_stain = np.arange(0, 10, 1)
    x_oil = np.arange(0, 10, 1)
    X, Y = np.meshgrid(x_stain, x_oil)
    Z = np.zeros_like(X)

    # Calculate washing time for each (sludge, grease) pair
    for i in range(len(x_stain)):
        for j in range(len(x_oil)):
            Z[i, j] = run_fuzzy_control_system(x_stain[i], x_oil[j], _plot=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Sludge')
    ax.set_ylabel('Grease')
    ax.set_zlabel('Washing Time')
    plt.show()

if __name__ == "__main__":
    plot_3d()
