import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
from Functions import *
import time

def Create_Models(par):    
    #### First model
    m1 = gp.Model("First")
    m1.params.outputFlag=0
     
    w = m1.addVars( par.num_features, par.num_classes, vtype=GRB.CONTINUOUS, lb=par.LB, ub=par.UB, name="w")
    m1.update()
 
    #Constraints

    #### Second model
    m2=[]; z = []
    for p in range(par.split_number):
        m2.append( gp.Model("Second[%s]"%p) )
        m2[p].params.outputFlag=0
    for p in range(par.split_number):
        z.append(  m2[p].addVars(par.num_features, par.num_classes, vtype=GRB.CONTINUOUS, lb=par.LB, ub=par.UB, name="z[%s]"%p) )
        m2[p].update()
        

        #Constraints

    return m1, m2, w, z

def First_Block_Problem(par, m1, w):  
    start = time.time()   
    #Objective Function
    Objective=0
    for p in range(par.split_number):
        for j in range(par.num_features):
            for k in range(par.num_classes):         
                Objective +=  par.Lambdas_val[p][j][k] * w[j,k] + 0.5*par.rho*( w[j,k] - par.Z_val[p][j][k] )*( w[j,k] - par.Z_val[p][j][k] ) 
    end = time.time()               
    # print("end-start=", end-start)
    
    # print("Objective=",Objective)
    m1.setObjective(Objective, GRB.MINIMIZE)
    m1.optimize()
    # print("m1.objval=", m1.objval)

    ## Update W_val
    for j in range(par.num_features):
        for k in range(par.num_classes):            
            # print("W_var=", par.W_val[j][k], " w=", m1.getVarByName("w[%s,%s]"%(j,k)).x)
            par.W_val[j][k] = m1.getVarByName("w[%s,%s]"%(j,k)).x
    
    return par, m1.Runtime

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

def Proximal_Second_Block_Problem(par, m2, z, x_train_agent, y_train_agent, Iteration):       

    tilde_xi = np.zeros((par.split_number, par.num_features, par.num_classes))    
    
    Total_runtime = 0
    for p in range(par.split_number):                
        
        ## Hypothesis
        H=calculate_hypothesis(par.Z_val[p], x_train_agent[p]) ## I_p x K matrix
        ## one-hot vector
        num_data = y_train_agent[p].shape[0]        
        y_train_Bin = np.zeros((num_data, par.num_classes)) 
        for idx in range(num_data):
            y_train_Bin[idx][y_train_agent[p][idx]] = 1.
        ## Gradient
        grad = calculate_gradient(par, par.total_data, H, x_train_agent[p], y_train_Bin)
        # grad = calculate_gradient(par, num_data, H, x_train_agent[p], y_train_Bin)      
                
        # ######
        # Z = tf.Variable( tf.convert_to_tensor(par.Z_val[p], np.float32) )        
        # batch_x = tf.convert_to_tensor(x_train_agent[p])
        # batch_y = tf.convert_to_tensor(y_train_agent[p])
        
        # with tf.GradientTape() as g:            
        #     pred = logistic_regression(batch_x,Z)                        
        #     loss = cross_entropy(pred, batch_y, par.num_classes)

        # # Compute gradients.        
        # gradients = g.gradient(loss, [Z])
        # # print( gradients[0].numpy() )
        # grad = gradients[0].numpy()


        ## Objective Function
        Objective=0        
        for j in range(par.num_features):
            for k in range(par.num_classes):
                
                Objective += grad[j][k] * z[p][j,k] + 0.5*par.rho*( par.W_val[j][k] - z[p][j,k] + (1.0/par.rho)*(par.Lambdas_val[p][j][k]- tilde_xi[p][j][k]) )*( par.W_val[j][k] - z[p][j,k] + (1.0/par.rho)*(par.Lambdas_val[p][j][k]- tilde_xi[p][j][k]) ) + (0.5/par.eta)*(z[p][j,k] - par.Z_val[p][j][k])*(z[p][j,k] - par.Z_val[p][j][k])

        # print("Objective=",Objective)
        
        m2[p].setObjective(Objective, GRB.MINIMIZE)
        m2[p].optimize()
        Total_runtime += m2[p].Runtime 
        # print("objval[%s]="%p, m2[p].objval)
        ## Update Z_val
        for j in range(par.num_features):
            for k in range(par.num_classes):
                # print("Z_var=", par.Z_val[p][j][k], " z=", m2[p].getVarByName("z[%s][%s,%s]"%(p,j,k)).x )
                par.Z_val[p][j][k] = m2[p].getVarByName("z[%s][%s,%s]"%(p,j,k)).x    
                
    return par, Total_runtime

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

def Trust_Second_Block_Problem(par, m2, z, x_train_agent, y_train_agent, Iteration):       

    tilde_xi = np.zeros((par.split_number, par.num_features, par.num_classes))    
    Total_runtime = 0
    for p in range(par.split_number):                
        ## Hypothesis
        H=calculate_hypothesis(par.Z_val[p], x_train_agent[p]) ## I_p x K matrix
        ## one-hot vector
        num_data = y_train_agent[p].shape[0]        
        y_train_Bin = np.zeros((num_data, par.num_classes)) 
        for idx in range(num_data):
            y_train_Bin[idx][y_train_agent[p][idx]] = 1.
        ## Gradient
        grad = calculate_gradient(par, par.total_data, H, x_train_agent[p], y_train_Bin)                       
        # grad = calculate_gradient(par, num_data, H, x_train_agent[p], y_train_Bin)         
        ## Objective Function
        Objective=0        
        for j in range(par.num_features):
            for k in range(par.num_classes):
                
                Objective += grad[j][k] * z[p][j,k] + 0.5*par.rho*( par.W_val[j][k] - z[p][j,k] + (1.0/par.rho)*(par.Lambdas_val[p][j][k]- tilde_xi[p][j][k]) )*( par.W_val[j][k] - z[p][j,k] + (1.0/par.rho)*(par.Lambdas_val[p][j][k]- tilde_xi[p][j][k]) ) 

        # print("Objective=",Objective)
        
        m2[p].setObjective(Objective, GRB.MINIMIZE)
        
        ## Trust-region
        LE = 0
        for j in range(par.num_features):
            for k in range(par.num_classes):
                LE += (z[p][j,k] - par.Z_val[p][j][k])*(z[p][j,k] - par.Z_val[p][j][k])
        trust = m2[p].addConstr(LE <= par.eta, name="LE")
        m2[p].optimize()
        Total_runtime += m2[p].Runtime 
        ## Remove        
        m2[p].remove(trust)

        # print("objval[%s]="%p, m2[p].objval)
        ## Update Z_val
        for j in range(par.num_features):
            for k in range(par.num_classes):
                # print("Z_var=", par.Z_val[p][j][k], " z=", m2[p].getVarByName("z[%s][%s,%s]"%(p,j,k)).x )
                par.Z_val[p][j][k] = m2[p].getVarByName("z[%s][%s,%s]"%(p,j,k)).x    
                
    return par, Total_runtime    

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


# def _Second_Block_Problem(par, m2, z, x_train_agent, y_train_agent, Iteration):   
#     #### Construct the gradient
#     grad=[]
#     grad = Construct_Gradient(par, x_train_agent, y_train_agent, grad)
 
#     #### Construct the models  
#     for p in range(par.split_number):

#         #Objective Function
#         Objective=0        
#         for j in range(par.parameter_size):
#             Objective += grad[p][j] * z[p][j] + 0.5*par.rho*( par.W_val[j] - z[p][j] + (1.0/par.rho)*(par.Lambdas_val[p][j]-par.tilde_xi[Iteration][p][j] ) )*( par.W_val[j] - z[p][j] + (1.0/par.rho)*(par.Lambdas_val[p][j]-par.tilde_xi[Iteration][p][j] ) ) #+ (0.5/par.eta)*(z[p][j] - par.Z_val[p][j])*(z[p][j] - par.Z_val[p][j])
            

#         # print("Objective=",Objective)
#         m2[p].setObjective(Objective, GRB.MINIMIZE)
 
#         ## Trust-region
#         LE = 0
#         for j in range(par.parameter_size):
#             LE += (z[p][j] - par.Z_val[p][j])*(z[p][j] - par.Z_val[p][j])
#         trust = m2[p].addConstr(LE <= par.eta, name="LE")
#         m2[p].optimize()
#         ## Remove        
#         m2[p].remove(trust)
        
#         ## Update Z_val
#         for j in range(par.parameter_size):
#             # print("Z_var=", par.Z_val[p][j], " z=", m2[p].getVarByName("z[%s][%s]"%(p,j)).x )
#             par.Z_val[p][j] = m2[p].getVarByName("z[%s][%s]"%(p,j)).x     
     
#     return par

 
