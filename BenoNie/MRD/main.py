import torch
from torch import nn
from sklearn.linear_model import LassoCV, ElasticNetCV
import numpy as np

import os
import sys
import pandas as pd
from utils import hrt_gauss,generate_conditional_data, set_seed,loss_f
from LassoNN import LassoNN
from MRD.data.DataGenerator import DataGenerator

from lasso_admm import lasso_admm
from sklearn.preprocessing import StandardScaler

    


def experiment_real(c,seed):
    p_vals_mrd_lasso = []
    p_vals_lasso = []
    p_vals_enet = []
    p_vals_mrd_enet = []

    real_data_gen = DataGenerator()
    X_mu, X_Sigma, (X_train, Y_train, X_test, Y_test) = real_data_gen.process_real_data(seed=seed, train_test_ratio=0.5)

    p = X_train.shape[1]

    scaler_Y = StandardScaler().fit(Y_train)
    Y_train = scaler_Y.transform(Y_train)
    Y_test = scaler_Y.transform(Y_test)

    # util model
    m=LassoCV(fit_intercept=False).fit(X_train,Y_train.ravel())

    #     fit Lasso
    lasso=LassoCV(fit_intercept=False).fit(X_train,Y_train.ravel())
    #     #fit Ours
    alpha_=lasso.alpha_
    mse_lasso=lasso.mse_path_[lasso.alphas_==lasso.alpha_].mean()

    T_coef_ = min(0.8,0.75*mse_lasso)
    zT,_=lasso_admm(X_train,Y_train,X_mu,X_Sigma, alpha_,T_coef=T_coef_,rho=1.,rel_par=1.,
                     MAX_ITER=150,ABSTOL=5e-4,RELTOL= 1e-3,is_NN=True,scaler=None)


    elnet = ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95],fit_intercept=False).fit(X_train, Y_train.ravel())
    alpha_ = elnet.alpha_ * elnet.l1_ratio_
    lmda_ = elnet.alpha_ * (1 - elnet.l1_ratio_)
    mse_elnet = elnet.mse_path_[elnet.alphas_ == elnet.alpha_].mean()
    # our Enet
    T_coef_ = min(0.8, .75 * mse_elnet)
    zT_enet, _ = lasso_admm(X_train, Y_train, X_mu, X_Sigma, alpha_, T_coef=T_coef_, rho=1., rel_par=1., \
                            MAX_ITER=150, ABSTOL=5e-4, RELTOL=1e-3, is_NN=True, l2_lmbda=lmda_, scaler=None)

    for j in range(p):


        m.coef_=zT
        t=((m.predict(X_test).ravel() - Y_test.ravel())**2).mean()
        p_val_MRD_lasso=hrt_gauss(m, t, X_test, Y_test, j, mu=X_mu, Sigma=X_Sigma,
                            is_iid=False, K=1000, is_NN=False,scaler=None)
        p_vals_mrd_lasso.append(p_val_MRD_lasso)

        m.coef_ = zT_enet
        t = ((m.predict(X_test).ravel() - Y_test.ravel()) ** 2).mean()
        p_val_MRD_enet = hrt_gauss(m, t, X_test, Y_test, j, mu=X_mu, Sigma=X_Sigma,
                             is_iid=False, K=1000, is_NN=False, scaler=None)
        p_vals_mrd_enet.append(p_val_MRD_enet)

        m.coef_ = elnet.coef_
        t = ((m.predict(X_test).ravel() - Y_test.ravel()) ** 2).mean()
        p_val_enet = hrt_gauss(m, t, X_test, Y_test, j, mu=X_mu, Sigma=X_Sigma,
                            is_iid=False, K=1000, is_NN=False, scaler=None)
        p_vals_enet.append(p_val_enet)

        m.coef_=lasso.coef_
        t=((m.predict(X_test).ravel() - Y_test.ravel())**2).mean()
        p_val_lasso=hrt_gauss(m, t, X_test, Y_test, j, mu=X_mu, Sigma=X_Sigma,
                                 is_iid=False, K=1000, is_NN=False,scaler=None)
        p_vals_lasso.append(p_val_lasso)

    result = pd.DataFrame({'c': [c],
                           'seed': [seed],
                           'lasso':[lasso.coef_],
                           'lasso_T':[zT],
                           'elnet': [elnet.coef_],
                           'elnet_T': [zT_enet],
                           'p_vals_0':[p_vals_lasso],
                           'p_vals_T':[p_vals_mrd_lasso],
                           'p_vals_E':[p_vals_enet],
                           'p_vals_ET':[p_vals_mrd_enet]
                           })

    return result


    
def experiment_NN(c,seed,is_linear=False, lr=5e-3, EPOCHS=60, is_est=False):
    # Set seed for reproducibility
    set_seed(seed)

    data_gen = DataGenerator(n=800, p=100)
    ones, X_mu, X_Sigma, (X_train, Y_train, X_test, Y_test) = data_gen.generate_AR1_data(c, rho=0.25, sparsity=0.3,
                                                                                         is_linear=is_linear, type='Poly',
                                                                                         train_test_ratio=0.5,
                                                                                         is_est=is_est)
    
    scaler_X=StandardScaler().fit(X_train)
    scaler_Y=StandardScaler().fit(Y_train)
    X_train, Y_train =scaler_X.transform(X_train), scaler_Y.transform(Y_train)
    X_test, Y_test =scaler_X.transform(X_test), scaler_Y.transform(Y_test)

    # NN
    m0 = LassoCV().fit(X_train, Y_train.ravel())
    l=nn.MSELoss()
    scaler=scaler_X
    alpha=m0.alpha_
    ftr_ = None
    n,p = X_train.shape

    y=Y_train
    X=X_train
    lr=lr
    num_epochs = EPOCHS
    ############# Train NN ################
    model_0=LassoNN(p,is_linear=is_linear)
    optimizer = torch.optim.Adam(model_0.parameters(), lr=lr)
    T_coef = 0
    for epoch in range(num_epochs):

        y_hat = model_0(torch.tensor(X).type(torch.FloatTensor))
        y_tensor = torch.tensor(y).type(torch.FloatTensor)
        loss = loss_f(y_hat,y_tensor,model_0,alpha,T_coef=T_coef)
        optimizer.zero_grad()  # zero the parameter gradients
        loss.backward()  # backpropagation
        optimizer.step()


    
    model_0.eval()
    mse_NN = ((model_0.predict(X_test).ravel() - Y_test.ravel())**2).mean()
    
###############################3######### extracting T_coef ##############################        
    model_dummy=LassoNN(p,is_linear=is_linear)

    optimizer = torch.optim.Adam(model_dummy.parameters(), lr=lr)
    for epoch in range(num_epochs):
       y_hat = model_dummy(torch.tensor(X[:int(0.8*n)]).type(torch.FloatTensor))
       y_tensor = torch.tensor(y[:int(0.8*n)]).type(torch.FloatTensor)
       loss = loss_f(y_hat,y_tensor,model_dummy,alpha,T_coef=0)
       optimizer.zero_grad()  # zero the parameter gradients
       loss.backward()  # backpropagation
       optimizer.step()

    model_dummy.eval()
    mse_ = ((model_dummy.predict(X[int(0.8*n):]).ravel() - y[int(0.8*n):].ravel())**2).mean()
    model_dummy.train()
    T_coef= min(0.8,0.75*mse_)

##########################################################################################
########### Train MRD NN################

    model=LassoNN(p,is_linear=is_linear)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):

        ftrs = np.random.permutation(range(p))[:int(0.25*p)] if ftr_ is None else [ftr_]
        if T_coef != 0:
          if scaler is not None:
              X_tilda_all = generate_conditional_data(scaler.inverse_transform(X), X_mu, X_Sigma,ftrs_=ftrs)
          else:
              X_tilda_all = generate_conditional_data(X, X_mu, X_Sigma,ftrs_=ftrs)

        y_hat = model(torch.tensor(X).type(torch.FloatTensor))
        y_tensor = torch.tensor(y).type(torch.FloatTensor)
        T_tilda = []
        T = l(y_tensor, y_hat) * torch.ones(len(ftrs))
        if T_coef != 0:
          for j in ftrs:#range(p):
              x_tilda = X.copy()
              if scaler is not None:
                x_tilda = scaler.inverse_transform(x_tilda)
                x_tilda[:, j] = X_tilda_all[:, j].copy()
                x_tilda = scaler.transform(x_tilda)
              else:
                x_tilda[:, j] = X_tilda_all[:, j].copy()
              y_hat_tilda = model(torch.tensor(x_tilda).type(torch.FloatTensor))
              T_tilda.append(l(y_hat_tilda, y_tensor))
          T_tilda = torch.stack(T_tilda)
        else:
          T_tilda = T

        loss = loss_f(y_hat,y_tensor,model,alpha,T,T_tilda,T_coef=T_coef)
        optimizer.zero_grad()  # zero the parameter gradients
        loss.backward()  # backpropagation
        optimizer.step()




    model.eval()
    mse_NN_T = ((model.predict(X_test).ravel() - Y_test.ravel())**2).mean()
    p_vals_NN=[];
    p_vals_MRD_NN = []
    for j in range(p):
        model_0.eval()
        t=((model_0.predict(X_test).ravel()-Y_test.ravel())**2).mean()
        p_val_NN=hrt_gauss(model_0, t, X_test, Y_test, j, mu=X_mu, Sigma=X_Sigma,
                        is_iid=False, K=1000, is_NN=True,scaler=scaler_X)
        model.eval()
        t=((model.predict(X_test).ravel()-Y_test.ravel())**2).mean()
        p_val_MRD_NN=hrt_gauss(model, t, X_test, Y_test, j, mu=X_mu, Sigma=X_Sigma,
                        is_iid=False, K=1000, is_NN=True,scaler=scaler_X)


        p_vals_NN.append(p_val_NN)
        p_vals_MRD_NN.append(p_val_MRD_NN)

    result = pd.DataFrame({'c': [c],
                           'seed': [seed],
                           'ones': [ones],
                           'mse_NN_T': [mse_NN_T],
                           'mse_NN': [mse_NN],
                           'p_vals_NN_T':[p_vals_MRD_NN],
                           'p_vals_NN':[p_vals_NN]
                           })
    return result


def experiment(c,seed,rho,lmbda=None,is_linear=True,is_est=False,GMM=False):
    # Set seed for reproducibility
    set_seed(seed)
    p_vals_mrd_lasso = []
    p_vals_lasso = []
    p_vals_enet = []
    p_vals_mrd_enet = []
    p_vals_ridge = []

    data_gen = DataGenerator(n=800, p=100)
    if GMM:
        ones, X_mu, X_Sigma, (X_train, Y_train, X_test, Y_test) = data_gen.generate_GMM_data(c, rhos=[0.1,0.2,0.3],
                                                                                             sparsity=0.3,
                                                                                             is_linear=is_linear,
                                                                                             type='Poly',
                                                                                             train_test_ratio=0.5)
    else:
        ones, X_mu, X_Sigma, (X_train, Y_train, X_test, Y_test) = data_gen.generate_AR1_data(c, rho=0.25, sparsity=0.3,
                                                                                             is_linear=is_linear,
                                                                                             type='Poly',
                                                                                             train_test_ratio=0.5,
                                                                                             is_est=is_est)


    scaler_X=StandardScaler().fit(X_train)
    scaler_Y=StandardScaler().fit(Y_train)
    X_train, Y_train =scaler_X.transform(X_train), scaler_Y.transform(Y_train)
    X_test, Y_test =scaler_X.transform(X_test), scaler_Y.transform(Y_test)

    rng = range(X_train.shape[1])

    #util model
    m = LassoCV(fit_intercept=False).fit(X_train, Y_train.ravel())

    #fit models

    # Enet
    elnet = ElasticNetCV(l1_ratio = [.1, .3, .5, .7, .9, .95],fit_intercept=False).fit(X_train,Y_train.ravel())

    # Ridge
    # ridge = RidgeCV(fit_intercept=False).fit(X_train,Y_train.ravel())

    # our Enet
    mse_elnet = elnet.mse_path_[elnet.alphas_==elnet.alpha_].mean()
    T_coef_=min(0.8,.75*mse_elnet) if lmbda is None else lmbda
    alpha_=elnet.alpha_*elnet.l1_ratio_
    lmda_=elnet.alpha_*(1-elnet.l1_ratio_)
    zT_enet,_=lasso_admm(X_train,Y_train,X_mu,X_Sigma,1*alpha_,T_coef=T_coef_,rho=rho,rel_par=1.,QUIET=True,\
                          MAX_ITER=150,ABSTOL=5e-4,RELTOL= 1e-3,is_NN=True,l2_lmbda=lmda_,scaler=scaler_X)

    # Lasso
    lasso=LassoCV(fit_intercept=False).fit(X_train,Y_train.ravel())

    #fit Ours
    alpha_=lasso.alpha_
    mse_lasso=lasso.mse_path_[lasso.alphas_==lasso.alpha_].mean()
    T_coef_=min(0.8,.75*mse_lasso) if lmbda is None else lmbda
    zT_lasso,_=lasso_admm(X_train,Y_train,X_mu,X_Sigma,1*alpha_,T_coef=T_coef_,rho=rho,rel_par=1.,QUIET=True,\
                          MAX_ITER=150,ABSTOL=5e-4,RELTOL= 1e-3,is_NN=True,scaler=scaler_X)


    for j in rng:

        #Lasso
        m.coef_ = lasso.coef_
        t = ((m.predict(X_test) - Y_test.ravel()) ** 2).mean()
        p_val_lasso = hrt_gauss(m, t, X_test, Y_test, j, mu=X_mu, Sigma=X_Sigma,
                               K=1000, is_NN=False, scaler=scaler_X)
        p_vals_lasso.append(p_val_lasso)

        #Lasso + our
        m.coef_ = zT_lasso
        t = ((m.predict(X_test) - Y_test.ravel()) ** 2).mean()
        p_val_mrd_lasso = hrt_gauss(m, t, X_test, Y_test, j, mu=X_mu, Sigma=X_Sigma,
                               K=1000, is_NN=False, scaler=scaler_X)
        p_vals_mrd_lasso.append(p_val_mrd_lasso)

        #Enet
        m.coef_ = elnet.coef_
        t = ((m.predict(X_test) - Y_test.ravel()) ** 2).mean()
        p_val_enet = hrt_gauss(m, t, X_test, Y_test, j, mu=X_mu, Sigma=X_Sigma,
                               K=1000, is_NN=False, scaler=scaler_X)
        p_vals_enet.append(p_val_enet)

        #Enet + our
        m.coef_ = zT_enet
        t = ((m.predict(X_test) - Y_test.ravel()) ** 2).mean()
        p_val_mrd_enet = hrt_gauss(m, t, X_test, Y_test, j, mu=X_mu, Sigma=X_Sigma,
                                K=1000, is_NN=False, scaler=scaler_X)
        p_vals_mrd_enet.append(p_val_mrd_enet)
#        
# #        #Ridge
#         m.coef_ = ridge.coef_
#         t = ((m.predict(X_test) - Y_test.ravel()) ** 2).mean()
#         p_val_ridge = hrt_gauss(m, t, X_test, Y_test, j, mu=X_mu, Sigma=X_Sigma,
#                            is_iid=False, K=1000, is_NN=False, scaler=scaler_X)
#         p_vals_ridge.append(p_val_ridge)



    result = pd.DataFrame({'c': [c],
                           'seed': [seed],
                           'rho':[rho],
                           'lmbda':[lmbda],
                           'ones': [ones],
                           'lasso':[lasso.coef_],
                           'lasso_T':[zT_lasso],
                           'elnet':[elnet.coef_],
                           'elnet_T':[zT_enet],
                           'p_vals_R':[p_vals_ridge],
                           'p_vals_0':[p_vals_lasso],
                           'p_vals_T':[p_vals_mrd_lasso],
                           'p_vals_E':[p_vals_enet],
                           'p_vals_ET':[p_vals_mrd_enet]
                           })
    return result




if __name__ == '__main__':
    
    date = '''TODO: fill in the date (str)'''
    # Parameters
    c = 1 #float(sys.argv[1])
    seed = 1 #int(sys.argv[2])
    rho = 0.25 #float(sys.argv[3])
    lmbda = 1 #float(sys.argv[4])
    '''
    If you want to run it locally you can run -
    for c in [0.13,0.14,0.15,0.16,0.17,0.18]:
        for seed in range(100):
    '''
    for _ in range(1):
      prefix = '''TODO: fill in the prefix of a desired directory name'''
      if lmbda > 1:
        lmbda_ = None
      else:
        lmbda_ = lmbda
      # Output directory and filename
      out_dir = "./results/results" + prefix + date +'/'
      out_file = out_dir + "c_" + str(c) + "_seed_" + str(seed) + ".csv"
      if os.path.exists(out_file):
        print("The file: --" +out_file+ "-- exists")
        continue #exit()
      # Run experiment
      result = experiment(c,seed,rho,lmbda=lmbda_,is_linear=True,is_est=False,GMM=False)
      #result = experiment_NN(c,seed,is_linear=False, lr=5e-3, EPOCHS=60, is_est=False)
      #result = experiment_real(c,seed)

  
  
  
      # Print the result
      print(result)
      sys.stdout.flush()
  
      # Write the result to output file
      if not os.path.exists(out_dir):
          os.mkdir(out_dir)
      result.to_csv(out_file, index=False, float_format="%.4f")
      print("Updated summary of results on\n {}".format(out_file))
      sys.stdout.flush()






