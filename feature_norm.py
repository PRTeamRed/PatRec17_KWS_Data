##############################################################
#
#    Name : feature_norm.py
#    Description: This script normalize the feature vectors
#    Written By : Mili Biswas
#    Date : 24.04.2018
#
##############################################################


import numpy as np
import math


def fNormLinear(m):
    l_rec,l_dim = np.shape(m) 
    l_final = []
    for i in range(l_dim):
        l_col = m[:,i] 
        l_mu = (np.sum(l_col)/l_rec)
        l_temp = l_col-l_mu
        l_sigma = (math.sqrt(np.sum(l_temp*l_temp)/(l_rec-1)))
        l_final.append(list(l_temp/l_sigma))
       # print("sigma=",l_sigma,"mu=",l_mu)
    
    return np.array(l_final).T


def fNormNonLinear(m,r=0.5):
    l_rec,l_dim = np.shape(m) 
    l_final = []
    for i in range(l_dim):
        l_col = m[:,i] 
        l_mu = (np.sum(l_col)/l_rec)
        l_temp = l_col-l_mu
        l_sigma = (math.sqrt(np.sum(l_temp*l_temp)/(l_rec-1)))
        l_final.append(list(1/(1 + np.exp(l_temp/(r*l_sigma)))))
       # print("sigma=",l_sigma,"mu=",l_mu)
    
    return np.array(l_final).T


if __name__ == "__main__":
    
    l_m = np.array([[999,3456,45555,3],[1.5,1.5,1.5,1.5],[30000,4000,45555,3],[-2345,49,-456,3],[1.5999,2.345,1.5,1.5]])
	print(fNormNonLinear(l_m))
    print(fNormLinear(l_m))
    