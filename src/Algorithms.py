try:
    import cupy as np  # (activate this if GPU is used)
except ImportError:
    import numpy as np  # (activate this if CPU is used)

import math
import time
from Models import *
from Functions import *

def DP_IADMM(par, x_train_agent, y_train_agent, x_train_new, y_train_new, x_test, y_test, file1):
    ## Data Label Reformatting: Integer -> Binary (train_data) for calculating the objective function value
    y_train_Bin = (np.arange(y_train_new.max()+1) == y_train_new[...,None]).astype(int) ## I X K
    num_data = y_train_Bin.shape[0]
    
    ### [0] Initialization
    ## Variables
    par.W_val = np.zeros((par.num_features, par.num_classes))  ## Global model parameters
    par.Z_val = np.zeros((par.split_number, par.num_features, par.num_classes)); ## Local model parameters defined for every agent
    par.Lambdas_val = np.zeros((par.split_number, par.num_features, par.num_classes)); ## Dual variables

    ## Matrix Normal Distribution used for Gaussian Mechanism for "Base" algorithm
    par.M = np.zeros((par.num_features, par.num_classes))  
    par.U = np.zeros((par.num_features, par.num_features))  
    par.V = np.zeros((par.num_classes, par.num_classes))      
    
    start_time_initial = time.time()    
    title="Iter    TrainCost     TestAcc     Violation    Elapsed(s)     IterT(s)   Solve_1(s)   Solve_2(s)    GradT(s)   NoiseT(s)  AbsNoiseMag    Z_change     AdapRho \n"
    print(title)    
    file1.write(title)

    for iteration in range(par.training_steps + 1):   
        start_time_iter = time.time()    
        ### Hyperparameter Rho 
        hyperparameter_rho(par, iteration)  ## see Functions.py

        ### [1] First Block Problem        
        if par.Algorithm == "OutP":                        
            par, Runtime_1, Avg_Noise_Mag, z_change_mean, Noise_Time, Grad_Time  = Base_First_Block_Problem_ClosedForm(par, x_train_agent, y_train_agent, iteration) ## see Models.py            
        else:                    
            par, Runtime_1 = First_Block_Problem_ClosedForm(par) ## see Models.py   
        
        ### [2] Second Block Problem        
        if par.Algorithm == "OutP":
            par, Runtime_2 = Base_Second_Block_Problem_ClosedForm(par) ## see Models.py           
        else:
            par, Runtime_2, Avg_Noise_Mag, z_change_mean, Noise_Time, Grad_Time = Second_Block_Problem_ClosedForm(par, x_train_agent, y_train_agent, iteration) ## see Models.py 
        

        ### [3] Dual update
        par.Lambdas_val += par.rho*(par.W_val - par.Z_val)        
         
        end_time = time.time()
        iter_time = end_time - start_time_iter
        elapsed_time = end_time - start_time_initial  

        ### Display intermediat results
        if iteration % par.display_step == 0: 
            H = calculate_hypothesis(par.W_val, x_train_new) ## I x K matrix (see Functions.py)                        
            cost = calculate_cost(par, num_data, H, y_train_Bin)  ## Compute the objective function value (see Functions.py)
            accuracy_test = calculate_accuracy(par, par.W_val, x_test, y_test)   ## Compute testing accuracy (see Functions.py)        
            residual = calculate_residual(par) ## Compute concensus violation (see Functions.py)
            
            results = '%4d %12.6e %12.6e %12.6e %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12.6e %12.6e %12.6e \n' %(iteration, cost, accuracy_test, residual, elapsed_time, iter_time, Runtime_1,  Runtime_2, Grad_Time, Noise_Time, Avg_Noise_Mag, z_change_mean, par.rho)  
            print(results)          
            file1.write(results)            

    return par.W_val, cost, file1
 
