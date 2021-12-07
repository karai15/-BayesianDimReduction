class DimensionalityReduction:
    def __init__(self, D, M, sigma2_y, m_W, Sigma_W, m_mu, Sigma_mu):
        # dimension
        self.D = D  # dim(y)
        self.M = M  # dim(x)
        # prior
        self.sigma2_y = sigma2_y  # (1,1)
        self.m_W = m_W  # (M, D)
        self.Sigma_W = Sigma_W  # (M, M, D)
        self.m_mu = m_mu  # (D, 1)
        self.Sigma_mu = Sigma_mu  # (D,D)

    def test_func(self):
        return 0

    def VariationalInference(self, Y_obs, max_iter):
        test = 1
