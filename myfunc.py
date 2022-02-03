import numpy as np
import matplotlib.pyplot as plt

# 配列を列ベクトルに変形 (N,1)
def c_vec(array):
    return array.reshape(-1, 1)

# 配列を行ベクトルに変形 (1,N)
def r_vec(array):
    return array.reshape(1, -1)

# plot data
def plot_image(Y, N_show, title):
    """
    :param Y: (N, L, L)
    :param N_show:
    """
    # Observation
    fig, axes = plt.subplots(N_show, N_show, tight_layout=True)
    cnt = 0
    for i in range(N_show):
        for j in range(N_show):
            axes[i, j].imshow(Y[cnt, :, :], cmap="jet")
            cnt += 1
    plt.suptitle(title)