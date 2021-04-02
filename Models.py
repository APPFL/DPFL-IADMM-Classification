import numpy as np
import math
from Functions import *
import time
 
 
def First_Block_Problem_ClosedForm(par):     
    
    ## Update W_val
    start = time.time()   
    for j in range(par.num_features):
        for k in range(par.num_classes):            
            par.W_val[j][k] = 0
            min_point = 0
            for p in range(par.split_number):
                min_point += par.Z_val[p][j][k] -  par.Lambdas_val[p][j][k] / par.rho
            min_point = min_point / par.split_number
            if min_point <= par.LB:
                par.W_val[j][k] = par.LB
            elif min_point >= par.UB:
                par.W_val[j][k] = par.UB
            else:
                par.W_val[j][k] = min_point
            

    end = time.time() 

    return par, end-start
 
def Proximal_Second_Block_Problem_ClosedForm(par, x_train_agent, y_train_agent, Iteration):       

    start = time.time()   
    for p in range(par.split_number):                
        
        ## Hypothesis
        H=calculate_hypothesis(par.Z_val[p], x_train_agent[p]) ## I_p x K matrix
        ## one-hot vector
        num_data = y_train_agent[p].shape[0]        
        y_train_Bin = np.zeros((num_data, par.num_classes)) 
        for idx in range(num_data):
            y_train_Bin[idx][y_train_agent[p][idx]] = 1.
        ## Gradient
        # grad = calculate_gradient(par, num_data, H, x_train_agent[p], y_train_Bin)                       
        grad = calculate_gradient(par, par.total_data, H, x_train_agent[p], y_train_Bin)                       

        ## Generate Laplacian Noise
        tilde_xi = np.zeros((par.num_features, par.num_classes))  
        if par.bar_epsilon != "none":
            tilde_xi = generate_laplacian_noise(par, H, num_data, x_train_agent[p], y_train_Bin, tilde_xi)            
             
        ## Update Z_val
        for j in range(par.num_features):
            for k in range(par.num_classes):                            
                min_point = (1.0/(par.rho + (1.0/par.eta)))*(  -grad[j][k] + par.rho*par.W_val[j][k] + par.Lambdas_val[p][j][k]- tilde_xi[j][k] +(1.0/par.eta)*par.Z_val[p][j][k] )
                
                if min_point <= par.LB:
                    par.Z_val[p][j][k] = par.LB
                elif min_point >= par.UB:
                    par.Z_val[p][j][k] = par.UB
                else:
                    par.Z_val[p][j][k] = min_point
            
    end = time.time()                 
    return par, end-start, np.mean(np.absolute(tilde_xi))
 
def Trust_Second_Block_Problem_ClosedForm(par, x_train_agent, y_train_agent, Iteration):  

    start = time.time()   
    for p in range(par.split_number):                
        
        ## Hypothesis
        H=calculate_hypothesis(par.Z_val[p], x_train_agent[p]) ## I_p x K matrix
        ## one-hot vector
        num_data = y_train_agent[p].shape[0]        
        y_train_Bin = np.zeros((num_data, par.num_classes)) 
        for idx in range(num_data):
            y_train_Bin[idx][y_train_agent[p][idx]] = 1.
        ## Gradient                          
        # grad = calculate_gradient(par, num_data, H, x_train_agent[p], y_train_Bin)                       
        grad = calculate_gradient(par, par.total_data, H, x_train_agent[p], y_train_Bin)     
    
        ## Generate Laplacian Noise
        tilde_xi = np.zeros((par.num_features, par.num_classes))  
        if par.bar_epsilon != "none":
            tilde_xi = generate_laplacian_noise(par, H, num_data, x_train_agent[p], y_train_Bin, tilde_xi)            
            # print("tilde_xi=", tilde_xi)
                        
        
        ## Update Z_val
        for j in range(par.num_features):
            for k in range(par.num_classes):                            
                min_point = (1.0/par.rho)*( -grad[j][k] + par.rho*par.W_val[j][k] + par.Lambdas_val[p][j][k] - tilde_xi[j][k]  )
                LB = max( -par.eta + par.Z_val[p][j][k], par.LB)
                UB = min( par.eta + par.Z_val[p][j][k], par.UB)
                if min_point <= LB:
                    par.Z_val[p][j][k] = LB
                elif min_point >= UB:
                    par.Z_val[p][j][k] = UB
                else:
                    par.Z_val[p][j][k] = min_point
             
    end = time.time()                 
    return par, end-start, np.mean(np.absolute(tilde_xi))

def Huang_First_Block_Problem_ClosedForm(par, x_train_agent, y_train_agent, Iteration):       
    
    start = time.time()   
    for p in range(par.split_number):                
        
        ## Hypothesis
        H=calculate_hypothesis(par.Z_val[p], x_train_agent[p]) ## I_p x K matrix
        ## one-hot vector
        num_data = y_train_agent[p].shape[0]        
        y_train_Bin = np.zeros((num_data, par.num_classes)) 
        for idx in range(num_data):
            y_train_Bin[idx][y_train_agent[p][idx]] = 1.
        ## Gradient
        grad = calculate_gradient(par, num_data, H, x_train_agent[p], y_train_Bin)       

        ## Decide eta
        par.eta = 0.0
        par.eta = calculate_eta_Huang(par, num_data, Iteration)
                    
        ## Update Z_val
        for j in range(par.num_features):
            for k in range(par.num_classes):                            
                min_point = (1.0/(par.rho + (1.0/par.eta)))*( -grad[j][k] + par.rho*par.W_val[j][k] + par.Lambdas_val[p][j][k] +(1.0/par.eta)*par.Z_val[p][j][k] )
                par.Z_val[p][j][k] = min_point         

        ## Generate Matrix Normal Noise
        tilde_xi = np.zeros((par.num_features, par.num_classes))  
        if par.bar_epsilon != "none":            
            tilde_xi = generate_matrix_normal_noise(par, num_data, tilde_xi)            
            # print("tilde_xi=", tilde_xi)
            par.Z_val[p] += tilde_xi
                    
            
    end = time.time()                 
    return par, end-start, np.mean(np.absolute(tilde_xi))

def Huang_Second_Block_Problem_ClosedForm(par):     
    
    ## Update W_val
    start = time.time()   
    for j in range(par.num_features):
        for k in range(par.num_classes):            
            par.W_val[j][k] = 0
            min_point = 0
            for p in range(par.split_number):
                min_point += par.Z_val[p][j][k] -  par.Lambdas_val[p][j][k] / par.rho
            min_point = min_point / par.split_number
            par.W_val[j][k] = min_point
            
    end = time.time() 

    return par, end-start

