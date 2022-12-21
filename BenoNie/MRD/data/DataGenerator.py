import numpy as np
from scipy import sparse
import pandas as pd
from utils import set_seed
from sklearn.utils import shuffle

class DataGenerator:
    def __init__(self, n=800, p=100):
        self.n = n
        self.p = p

    def _create_sparse_vector(rows=1000, density=0.2, seed=0):
        S = sparse.random(m=rows, n=1, density=density, format='coo', dtype=None, random_state=seed, data_rvs=None)
        sparse_vec = S.A
        beta = [sparse_vec != 0][0].astype(int)
        return beta

    def _create_AR1_Sigma(p, rho=0.5):
        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = rho ** (abs(i - j))
        mu = np.zeros((p, 1)).ravel()
        return mu, Sigma

    def _create_normal_noise(mu, sigma, shape):
        return np.random.normal(mu, sigma, (shape[0], shape[1]))

    def _generate_Y(self, c, beta, X, is_linear=True, type='Poly'):
        v = self._create_normal_noise(mu=0, sigma=1, shape=(self.n, 1))
        if is_linear:
            Y = X @ beta + v
        elif type == 'Poly':
            Y = 0.5 * np.power(X @ beta, 3) + v
        else:
            raise NotImplementedError
        return Y

    def _split_data(self,X,Y,r=0.5):

        X_train = X[0:int(r * self.n)]
        Y_train = Y[0:int(r * self.n)]
        X_test = X[int(r * self.n):]
        Y_test = Y[int(r * self.n):]
        return X_train, Y_train, X_test, Y_test


    def generate_AR1_data(self, signal_strength, rho=0.25, sparsity=0.3, is_linear=True, type='Poly', train_test_ratio=0.5, is_est=False):
        c=signal_strength
        X_mu, X_Sigma = self._create_AR1_Sigma(self.p, rho=rho)
        X = np.random.multivariate_normal(X_mu, X_Sigma, self.n)
        ones = self._create_sparse_vector(self.p, sparsity)
        for i, one in enumerate(ones):
            if np.random.rand() > 0.5:
                ones[i] = -one
        beta = c * ones

        Y = self._generate_Y(self, c, beta, X, is_linear=is_linear, type=type)


        X_train, Y_train, X_test, Y_test = self._split_data(self,X,Y,r=train_test_ratio)

        if is_est:
            X_mu, X_Sigma = np.mean(X, axis=0), np.cov(X.T)

        return ones, X_mu, X_Sigma, (X_train, Y_train, X_test, Y_test)


    def generate_GMM_data(self, signal_strength, rhos=[0.1,0.2,0.3], sparsity=0.3, is_linear=True, type='Poly', train_test_ratio=0.5):
        c=signal_strength

        X = np.zeros((self.n, self.p))
        for sample in range(self.n):
            randnum = np.random.rand() * 3
            if randnum < 1:
                X_mu, X_Sigma = self._create_AR1_Sigma(self.p, rho=rhos[0])
                X[sample] = np.random.multivariate_normal(X_mu, X_Sigma, 1)
            elif randnum < 2:
                X_mu, X_Sigma = self._create_AR1_Sigma(self.p, rho=rhos[1])
                X[sample] = np.random.multivariate_normal(X_mu, X_Sigma, 1)
            else:
                X_mu, X_Sigma = self._create_AR1_Sigma(self.p, rho=rhos[2])
                X[sample] = np.random.multivariate_normal(X_mu, X_Sigma, 1)

        ones = self._create_sparse_vector(self.p, sparsity)
        for i, one in enumerate(ones):
            if np.random.rand() > 0.5:
                ones[i] = -one
        beta = c * ones

        Y = self._generate_Y(self, c, beta, X, is_linear=is_linear, type=type)

        X_train, Y_train, X_test, Y_test = self._split_data(self,X,Y,r=train_test_ratio)

        X_mu_est, X_Sigma_est = np.mean(X, axis=0), np.cov(X.T)

        return ones, X_mu_est, X_Sigma_est, (X_train, Y_train, X_test, Y_test)

    def process_real_data(self,seed=None,train_test_ratio=0.5):
        df = pd.read_csv('HIV.csv').dropna().to_numpy()
        set_seed(seed)

        X = df[:, 1:]
        Y = df[:, 0]

        X, Y = shuffle(X, Y, random_state=seed)
        Y = Y.reshape(-1, 1)
        X_mu_est, X_Sigma_est = np.mean(X, axis=0), np.cov(X.T)

        X_train, Y_train, X_test, Y_test = self._split_data(self,X,Y,r=train_test_ratio)

        return X_mu_est, X_Sigma_est, (X_train, Y_train, X_test, Y_test)



