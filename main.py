# coding=utf-8
from copy import deepcopy
import numpy as np
# import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from src.BayseDimensionalityReduction.DimensionalityReduction import *  # 自作Class wcResultClassのimport

###########################
# # # skip=2
# aaa = np.array([1, 2, 3, 4, 5], dtype="float32")
# # # aaa[1] /= 0
# # x = np.nan
# # aaa[1] = x
# # # bbb = aaa[0::skip]
# # # aaa = np.random.binomial(1, 0.5, size=(10, 2))
# dim = DimensionalityReduction()
# test = 1
###########################

def load_facedata(skip):
    """
    scikit-learnのサンプルデータセットの一覧と使い方
    https://note.nkmk.me/python-sklearn-datasets-load-fetch/
    """
    face = datasets.fetch_olivetti_faces()  # 同一人物の様々な状態の顔画像（40人 x 10枚）
    Y_raw = face["images"]  # (400, 64, 64)

    N, Sraw_row, Sraw_col = Y_raw.shape
    Y_tmp = Y_raw[:, 0::skip, 0::skip]  # slice data

    # # plot test
    # plt.figure()
    # plt.imshow(Y_tmp[1, :, :])
    # plt.show()

    # convert D dimensional vector * N set
    Y = Y_tmp.reshape(Y_tmp.shape[1] * Y_tmp.shape[2], N)
    D = Y.shape[0]  # Y of dimension

    return Y, D


def miss_facedata(Y, missing_rate):
    mask = np.random.binomial(1, missing_rate, size=Y.shape)  # 0,1
    Y_obs = deepcopy(Y)
    Y_obs[np.where(mask == 1)] = np.nan
    return Y_obs


def main():
    skip = 2
    missing_rate = 0.5
    Y, D = load_facedata(skip)
    Y_obs = miss_facedata(Y, missing_rate)

    # # param
    M = 16  # dimension of latent variable
    # prior
    sigma2_y = 0.001
    m_mu = np.zeros(D)
    Sigma_mu = np.eye(D)
    m_W = np.zeros((M, D))
    Sigma_W = np.zeros((M, M, D))  # (M, M) * D set
    for d in range(D):
        Sigma_W[:, :, d] = 0.1 * np.eye(M)

    # instance
    dimensionalityReduction = DimensionalityReduction(D, M, sigma2_y, m_W, Sigma_W, m_mu, Sigma_mu)

    # learn & generate
    max_iter = 100
    dimensionalityReduction.VariationalInference(deepcopy(Y_obs), max_iter)

    # DimensionalityReduction  97line

main()
test = 1
