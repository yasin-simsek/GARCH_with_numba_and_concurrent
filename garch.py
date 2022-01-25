# libraries
import numpy as np
import pandas as pd
from scipy.optimize import fmin_slsqp

# negative of loglikelihood of the garch model
def garch_negloglik(params, r):
    omega = params[0]
    beta = params[1]
    alpha = params[2]
    sigma2_0 = 1
    
    T = len(r)
    sigma2 = np.zeros(T,)
    sigma2[0] = sigma2_0
    
    logliks = np.zeros(T,)
    logliks[0] = -0.5*np.log(2*np.pi*sigma2[0]) - (r[0]**2)/(2*sigma2[0])
    
    for t in range(1,T):
        sigma2[t] = omega + beta*sigma2[t-1] + alpha*(r[t-1]**2)
        logliks[t] = -0.5*np.log(2*np.pi*sigma2[t]) - (r[t]**2)/(2*sigma2[t])
        
    negloglik = -logliks.sum()
    
    return negloglik  

# stationarity constraint
def garch_constraint(params, r):
    
    omega = params[0]
    beta = params[1]
    alpha = params[2]

    return np.array([1-alpha-beta])

# 1 step ahead forecast
def garch_forecast(r):
    # maximum likelihood estimation
    bounds = [(1e-6, 2*r.var()),(1e-6,0.999),(1e-6,0.999)] # upper-lower bounds 
    startingVals = np.array([0.15, 0.6, 0.3]) # initial values for optimization
    theta_hat = fmin_slsqp(garch_negloglik, startingVals,
                           f_ieqcons=garch_constraint, bounds = bounds, disp=None, iprint=0,
                           args = (r,)) # optimization

    # 1 step ahead forecast
    temp = theta_hat[0] + theta_hat[1]*(np.var(r)) + theta_hat[2]*r[0]
    for j in range(1,len(r)):
        temp = theta_hat[0] + theta_hat[1]*temp + theta_hat[2]*r[j]
    
    return temp 