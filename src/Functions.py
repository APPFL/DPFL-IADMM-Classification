try:
    import cupy as np  # (activate this if GPU is used)
except ImportError:
    import numpy as np  # (activate this if CPU is used)

import math
from scipy.stats import matrix_normal
from scipy.special import softmax
import time
 
def calculate_hypothesis(W_val, x_train):
    # FIXME: This may copy data between cpu and gpu. Consider torch.
    return np.array(softmax(np.matmul(x_train, W_val).get(), axis=1))

def calculate_cost(par, num_data, H, y_train_Bin):
    return -np.sum( np.multiply(y_train_Bin, np.log(H)) ) / num_data + par.gamma * np.sum( np.square(par.W_val) )

def calculate_accuracy(par, W, x_train, y_train):
    H = calculate_hypothesis(W, x_train)
    H_argmax = np.argmax(H, axis=1)        
    return np.mean( np.equal(H_argmax, y_train ) )

def calculate_gradient(par, num_data, H, x_train, y_train_Bin):
    grad = np.zeros((par.num_features, par.num_classes))    ## J X K matrix
    H_hat= np.subtract(H, y_train_Bin)    ## I X K matrix
    grad = np.matmul(x_train.transpose(), H_hat) / num_data + 2. * par.gamma * par.W_val ## J X K matrix
    
    return grad

def calculate_residual(par):
    Temp = np.absolute(par.W_val - par.Z_val)
    residual = np.sum(Temp) / par.split_number
    return residual

def generate_laplacian_noise(par, H, num_data, x_train, y_train_Bin, tilde_xi):
    
    H_hat=np.subtract(H, y_train_Bin)    ## I_p X K matrix
    H_hat_abs = np.absolute(H_hat)

    x_train_sum = np.sum(x_train, axis = 1)
    H_hat_abs_sum = np.sum(H_hat_abs, axis = 1)
    x_train_H_hat_abs = np.multiply(x_train_sum,H_hat_abs_sum) / num_data
    bar_lambda = np.max(x_train_H_hat_abs)/float(par.bar_eps_str)
    
    
    tilde_xi_shape = par.M + bar_lambda
    tilde_xi = np.random.laplace( par.M, tilde_xi_shape, [par.num_features, par.num_classes])
    
    return tilde_xi

def calculate_eta_Base(par,num_data, Iteration):
    
    delta = 1e-6  ## (epsilon, delta)-differential privacy
    c1 = num_data*1
    c3 = num_data*0.25
    cw = math.sqrt( par.num_features*par.num_classes*4 )
    
    if par.bar_eps_str != "infty":      
        par.eta = 1.0 / ( c3 + 4.0*c1*math.sqrt( par.num_features*par.num_classes*(Iteration+1)*math.log(1.25/delta)  )/(num_data*float(par.bar_eps_str)*cw)  )        
    else:
        par.eta = 1.0 / c3 
                
    par.eta = par.eta * float(par.a_str)

    return par.eta

def generate_matrix_normal_noise(par, num_data,tilde_xi):
    c1 = num_data*1
    delta = 1e-6  ## 1e-308, 1e-6

    sigma = 2*c1*math.sqrt(2*math.log(1.25/delta)) / (num_data*float(par.bar_eps_str)*(par.rho + 1.0/par.eta))    

    tilde_xi_shape = par.M + sigma*sigma  
    tilde_xi = np.random.normal( par.M, tilde_xi_shape, [par.num_features, par.num_classes])
    
    return tilde_xi

def hyperparameter_rho(par, iteration):

    if par.rho_str == "dynamic_1" or par.rho_str == "dynamic_2":
        if par.Instance =="MNIST":            
            c1 = 2.0; c2=5.0; Tc = 10000.0; rhoC=1.2
        elif par.Instance =="FEMNIST":
            c1 = 0.005; c2=0.05; Tc = 2000.0; rhoC=1.2
        elif par.Instance =="CIFAR10":
            c1 = 2.0; c2=5.0; Tc = 10000.0; rhoC=1.2
        else:
            raise AssertionError("Unexpected value of 'par.Instance'!", par.Instance)
            
        if par.bar_eps_str == "infty":
            par.rho = c1 * math.pow(rhoC, math.floor( (iteration+1) / Tc ) ) 
        else:
            par.rho = c1 * math.pow(rhoC, math.floor( (iteration+1) / Tc ) ) + c2/float(par.bar_eps_str) 
        if par.rho_str == "dynamic_2":
            par.rho = par.rho/100.0                
    else:
        par.rho = float(par.rho_str)         

    # the parameter is bounded above
    if par.rho > 1e9:
        par.rho = 1e9



 
        
            
