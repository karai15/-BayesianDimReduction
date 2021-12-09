from copy import deepcopy
import numpy as np
from src.BayseDimensionalityReduction.myfunc import *

class DimensionalityReduction:

    # # 配列を列ベクトルに変形 (N,1)
    # def c_vec(self, array):
    #     return array.reshape(-1, 1)
    #
    # # 配列を行ベクトルに変形 (1,N)
    # def r_vec(self, array):
    #     return array.reshape(1, -1)

    def initial(self, Y, prior):
        M = prior["M"]
        D, N = Y.shape
        X = np.random.normal(loc=0, scale=1, size=(M, N))  # <x>_p(x), p(x) = N(x|0, I_M)
        XX = np.zeros((M, M, N))  # <xx>_p(x)
        for n in range(N):
            XX[:, :, n] = c_vec(X[:, n]) @ r_vec(X[:, n]) + np.eye(M)  # <xx> = <x><x>^T + I_M
        return X, XX

    def interpolate(self, mask, X, posterior):
        N = X.shape[1]
        m_W = posterior["m_W"]
        m_mu = posterior["m_mu"]
        # update
        Y_est = m_W.T @ X + np.kron(np.ones((1, N)), c_vec(m_mu))  # Estimate all of Y
        return Y_est

    def update_W(self, Y, prior, posterior, X, XX):
        D = prior["D"]
        M = prior["M"]
        N = Y.shape[1]
        m_mu = posterior["m_mu"]

        # VI for q(W) = q(w1)*q(w2)* ... q(wD)
        m_W = np.zeros((M, D))
        Sigma_W = np.zeros((M, M, D))
        for d in range(D):
            Sigma_W[:, :, d] = np.linalg.inv(1 / prior["sigma2_y"] * np.sum(XX, axis=2) + np.linalg.inv(prior["Sigma_W"][:, :, d]))
            m_W[:, d] =  (1 / prior["sigma2_y"] * Sigma_W[:, :, d] @ X @ (c_vec(Y[d, :]) - m_mu[d] * np.ones((N, 1)) ))[:, 0]

        # update
        posterior["Sigma_W"] = Sigma_W
        posterior["m_W"] = m_W

        return posterior

    def update_mu(self, Y, prior, posterior, X, XX):
        D = prior["D"]
        M = prior["M"]
        N = Y.shape[1]
        m_W = posterior["m_W"]

        # VI for q(mu)
        Sigma_mu = np.linalg.inv(N / prior["sigma2_y"] * np.eye(D) + np.linalg.inv(prior["Sigma_mu"]))
        m_mu = 1 / prior["sigma2_y"] * Sigma_mu @ (Y - m_W.T @ X ) @ np.ones(N)

        # update
        posterior["Sigma_mu"] = Sigma_mu
        posterior["m_mu"] = m_mu

        return posterior

    def update_X(self, Y, prior, posterior):
        D = posterior["D"]
        M = posterior["M"]
        N = Y.shape[1]
        m_mu = posterior["m_mu"]
        m_W = posterior["m_W"]
        Sigma_W = posterior["Sigma_W"]

        # VI for q(X) = q(x1)*q(x2)* ... * q(xN)
        X = np.zeros((M, N))
        XX = np.zeros((M, M, N))
        for n in range(N):
            Sigma_X = np.linalg.inv(1 / prior["sigma2_y"] * (m_W @ m_W.T + np.sum(Sigma_W, axis=2)) + np.eye(M))
            mu_X = 1 / prior["sigma2_y"] * Sigma_X @ m_W @ (Y[:, n] - m_mu)
            X[:, n] = mu_X
            XX[:, :, n] = c_vec(mu_X) @ r_vec(mu_X) + Sigma_X

        return X, XX

    def VariationalInference(self, Y, prior, max_iter):
        """
        :param Y:
        :param prior: {D, M, sigma2_y, m_mu, Sigma_mu, m_W, Sigma_W}
        :param max_iter:
        :return:
        """

        # initial value
        X, XX = self.initial(Y, prior)  # X(M, N), XX(M, M, N)
        mask = np.isnan(Y)
        sum_mask = np.sum(mask)
        posterior = deepcopy(prior)  # {D, M, sigma2_y, m_mu, Sigma_mu, m_W, Sigma_W}

        # VI
        for iter in range(max_iter):
            print("iter=", iter)

            # Interpolate
            if sum_mask > 0:
                Y_est = self.interpolate(mask, X, posterior)  # Estimate all of data Y
                Y[np.where(mask == True)] = Y_est[np.where(mask == True)]  # Interpolate a part of Y (=Y[mask])

            # M-step (Estimate parameter {W, mu})
            posterior = self.update_W(Y, prior, posterior, X, XX)
            posterior = self.update_mu(Y, prior, posterior, X, XX)

            # E-step (Estimate latent variable)
            X, XX = self.update_X(Y, prior, posterior)

        return posterior, X
