# coding=utf-8
from copy import deepcopy
import numpy as np
import time
import sklearn.datasets as datasets
from src.BayseDimensionalityReduction.DimensionalityReduction import *
from src.BayseDimensionalityReduction.myfunc import *

def load_facedata(skip):
    """
    scikit-learnのサンプルデータセットの一覧と使い方
    https://note.nkmk.me/python-sklearn-datasets-load-fetch/
    """
    face = datasets.fetch_olivetti_faces()  # 同一人物の様々な状態の顔画像（40人 x 10枚）
    Y_raw = face["images"]  # (400, 64, 64)
    N, Sraw_row, Sraw_col = Y_raw.shape
    L = np.round(Sraw_row / skip).astype("int")  # image height
    Y_tmp = Y_raw[:, 0::skip, 0::skip]  # slice data

    # convert D dimensional vector * N set
    Y = Y_tmp.reshape(Y_tmp.shape[1] * Y_tmp.shape[2], N)
    D = Y.shape[0]  # Y of dimension

    # # plot test
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(Y_tmp[0, :, :])
    # plt.show()

    return Y, D, L


def miss_facedata(Y, missing_rate):
    """
    :param Y:
    :param missing_rate:
    :return:
    """
    mask = np.random.binomial(1, missing_rate, size=Y.shape)  # 0,1

    ###############
    # _mask = np.random.binomial(1, missing_rate, size=Y.shape[0])  # 0,1
    # mask = c_vec(_mask) @ np.ones((1, Y.shape[1]), dtype=_mask.dtype)
    # mask = np.zeros(Y.shape)
    # mask[0:Y.shape[0]:4, 0:Y.shape[1]:4] = 1
    ###############

    Y_obs = deepcopy(Y)
    Y_obs[np.where(mask == 1)] = np.nan
    return Y_obs, mask


def main():
    start = time.time()
    skip = 2
    missing_rate = 0.3  # [0, 1]
    Y, D, L = load_facedata(skip)
    Y_obs, mask = miss_facedata(Y, missing_rate)
    N = Y_obs.shape[1]

    # # param
    M = 32  # dimension of latent variable
    # prior
    Sigma_W =  np.zeros((M, M, D))
    for d in range(D):
        Sigma_W[:, :, d] = 0.1 * np.eye(M)
    prior= {
        "D": D,
        "M": M,
        "sigma2_y": 0.001,
        "m_mu": np.zeros(D),  # <mu>_p(mu), p(mu)=N(mu|0, I_M)
        "Sigma_mu": np.eye(D),  # <mu*mu^T>_p(mu)
        "m_W": np.zeros((M, D)),  # <w_d>_p(w_d), p(w_d) = N(w_d|0, Sigma_W)
        "Sigma_W": Sigma_W  # <WW^T> (M, M) * D
    }

    # learn & generate
    max_iter = 10
    dimensionalityReduction = DimensionalityReduction()      # instance
    posterior, X_est = dimensionalityReduction.VariationalInference(deepcopy(Y_obs), prior, max_iter)
    Y_est = posterior["m_W"].T @ X_est + np.kron(np.ones((1, N)), c_vec(posterior["m_mu"]))
    Y_itp = deepcopy(Y_obs)  # interpolation data
    Y_itp[np.where(mask==1)] = Y_est[np.where(mask==1)]

    # convert Y (1dim => 2dim)
    Y_obs = Y_obs.reshape(N, L, L)
    Y_itp = Y_itp.reshape(N, L, L)
    Y_est = Y_est.reshape(N, L, L)
    Y_truth = Y.reshape(N, L, L)

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # plot
    N_show = 4
    plot_image(Y_obs, N_show, "Observation")  # Observation
    plot_image(Y_itp, N_show, "Interpolation")  # Interpolation
    plot_image(Y_est, N_show, "Estimation")  # Estimation
    plot_image(Y_truth, N_show, "Truth")  # Estimation
    plt.show()


if __name__ == "__main__":
    main()