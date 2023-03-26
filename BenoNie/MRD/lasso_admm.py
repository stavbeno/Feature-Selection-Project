import time
import torch
from torch import nn
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm, cholesky
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
mrd_idx = script_dir.find('/MRD')
mrd_dir = script_dir[:mrd_idx]
sys.path.append(mrd_dir + '/MRD')
from LassoNN import LassoNN
import utils_mrd




def lasso_admm(X, y, X_mu, X_Sigma, alpha, T_coef, rho=1., rel_par=1., QUIET=True,\
               MAX_ITER=50, ABSTOL=1e-3, RELTOL=1e-2, is_NN=True, ftr_=None, scaler=None, lr=8e-3, EPOCHS=35, l2_lmbda=0):
    '''
    Fit MRD-lasso using ADMM
    X: Training features\
    y: Training respone (labels)
    X_mu: The expectation vector of the features, i.e., np.mean(X, axis=0)
    X_Sigma: The covariance matrix of the features, i.e.,  np.cov(X.T)
    T_coef: The MRD penalty parameter (\lambda in the paper)
    is_NN: Using Pytorch. Must be true if T_coef > 0.
    ftr_: is not None (int), then optimizing (MRD) for the specific given feature. 
    scaler: The scaler of the features. Should be in the form of Sklearn.
    lr: learning rate for the Pytorch mechanism.
    EPOCHS: number of epochs for the Pytorch mechanism.
    All others inputs are for the ADMM procedure.
    '''
    
    if T_coef < 0 or T_coef > 1:
        raise ValueError("lambda should be between 0 to 1")
    
    if T_coef>0:
        assert is_NN, 'is_NN must be true if T_coef >0'
    
    alpha = (1 - T_coef) * alpha

    if not QUIET:
        tic = time.time()

    # Data preprocessing
    m, n = X.shape
    p = n
    # save a matrix-vector multiply
    Xty = X.T.dot(y)

    # ADMM solver
    x = np.zeros((n, 1))
    z = np.zeros((n, 1))
    u = np.zeros((n, 1))

    # cache the (Cholesky) factorization
    L, U = factor(X, rho)

    if not QUIET:
        print('\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s' % ('iter',
                                                       'r norm',
                                                       'eps pri',
                                                       's norm',
                                                       'eps dual',
                                                       'objective'))

    # Saving state
    h = {}
    h['objval'] = np.zeros(MAX_ITER)
    h['r_norm'] = np.zeros(MAX_ITER)
    h['s_norm'] = np.zeros(MAX_ITER)
    h['eps_pri'] = np.zeros(MAX_ITER)
    h['eps_dual'] = np.zeros(MAX_ITER)
    for k in range(MAX_ITER):

        # x-update
        if is_NN:
            l = nn.MSELoss()
            model = LassoNN(p)
            model.fc_layer[0]._parameters['weight'] = \
                (torch.tensor(x).type(torch.FloatTensor)).clone().detach().T
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
            for epoch in range(EPOCHS):

                ftrs = np.random.permutation(range(p))[:int(.25 * p)] if ftr_ is None else [ftr_]
                if T_coef != 0:
                    if scaler is not None:
                        X_tilda_all = utils_mrd.generate_conditional_data(scaler.inverse_transform(X), X_mu, X_Sigma, ftrs_=ftrs)
                    else:
                        X_tilda_all = utils_mrd.generate_conditional_data(X, X_mu, X_Sigma, ftrs_=ftrs)

                y_hat = model(torch.tensor(X).type(torch.FloatTensor))
                z_tensor = torch.tensor(z.T).type(torch.FloatTensor)
                u_tensor = torch.tensor(u.T).type(torch.FloatTensor)
                y_tensor = torch.tensor(y).type(torch.FloatTensor)
                T_tilda = []
                T = l(y_tensor, y_hat) * torch.ones(len(ftrs))
                if T_coef != 0:
                    for j in ftrs:  # range(p):
                        x_tilda = X.copy()
                        if scaler is not None:
                            x_tilda = scaler.inverse_transform(x_tilda)
                            x_tilda[:, j] = X_tilda_all[:, j].copy()
                            x_tilda = scaler.transform(x_tilda)
                        else:
                            x_tilda[:, j] = X_tilda_all[:, j].copy()
                        y_hat_tilda = model(torch.tensor(x_tilda).type(torch.FloatTensor))
                        T_tilda.append(l(y_hat_tilda, y_tensor))
                    T_tilda = torch.tensor(T_tilda).requires_grad_()
                else:
                    T_tilda = T
                loss = loss_admm(y_hat, y_tensor, rho, model, z_tensor, u_tensor, T, T_tilda, T_coef=T_coef)
                optimizer.zero_grad()  # zero the parameter gradients
                loss.backward()  # backpropagation
                optimizer.step()
            x=model.fc_layer[0]._parameters['weight'].T.clone().detach().numpy()
  
        else:
            q = Xty + rho * (z - u)  # (temporary value)
            if m >= n:
                x = spsolve(U, spsolve(L, q))[..., np.newaxis]
            else:
                ULXq = spsolve(U, spsolve(L, X.dot(q)))[..., np.newaxis]
                x = (q * 1. / rho) - ((X.T.dot(ULXq)) * 1. / (rho ** 2))

        # z-update with relaxation
        zold = np.copy(z)
        x_hat = rel_par * x + (1. - rel_par) * zold
        z = shrinkage(x_hat + u, alpha * 1. / rho) / (1 + l2_lmbda)

        # u-update
        u += (x_hat - z)

        # diagnostics, reporting, termination checks
        h['objval'][k] = objective(X, y, alpha, x, z)
        h['r_norm'][k] = norm(x - z)
        h['s_norm'][k] = norm(-rho * (z - zold))
        h['eps_pri'][k] = np.sqrt(n) * ABSTOL + \
                          RELTOL * np.maximum(norm(x), norm(-z))
        h['eps_dual'][k] = np.sqrt(n) * ABSTOL + \
                           RELTOL * norm(rho * u)
        if not QUIET:
            print('%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' % (k + 1, \
                                                                   h['r_norm'][k], \
                                                                   h['eps_pri'][k], \
                                                                   h['s_norm'][k], \
                                                                   h['eps_dual'][k], \
                                                                   h['objval'][k]))

        if (h['r_norm'][k] < h['eps_pri'][k]) and (h['s_norm'][k] < h['eps_dual'][k]):
            break

    if not QUIET:
        toc = time.time() - tic
        print("\nElapsed time is %.2f seconds" % toc)

    return z.ravel(), model



def objective(X,y,alpha,x,z):
    return .5*np.square(X.dot(x)-y).sum()+alpha*norm(z,1)

def shrinkage(x,kappa):
    return np.maximum(0.,x-kappa)-np.maximum(0.,-x-kappa)

def factor(X,rho):
    m,n = X.shape
    if m>=n:
       L = cholesky(X.T.dot(X)+rho*sparse.eye(n))
    else:
       L = cholesky(sparse.eye(m)+1./rho*(X.dot(X.T)))
    L = sparse.csc_matrix(L)
    U = sparse.csc_matrix(L.T)
    return L,U


def loss_admm(y_hat, y, rho, model, z, u, T=0, T_tilda=0, T_coef=0):
    l = nn.MSELoss()
    loss = (1 - T_coef) * 0.5 * l(y, y_hat) + \
           (rho / 2.) * (model.fc_layer[0]._parameters['weight'].requires_grad_() - z + u).norm(p=2).pow(2) + \
           T_coef * torch.sigmoid(T - T_tilda).mean()

    return loss
