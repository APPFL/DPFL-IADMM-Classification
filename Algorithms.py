import numpy as np
import math
import time
from Models import *
from Functions import *
import tensorflow as tf



def Centralized_SGD_TensorFlow(par, train_data, file1):
    ################################################
    ###### Logistic Regression using TensorFlow2 (see https://builtin.com/data-science/guide-logistic-regression-tensorflow-20 )
    ################################################
    # Weight of shape [784, 10], the 28*28 image features, and a total number of classes.
    W = tf.Variable(tf.ones([par.num_features, par.num_classes]), name="weight")

    # Bias of shape [10], the total number of classes.
    # b = tf.Variable(tf.zeros([num_classes]), name="bias")
    
    # Stochastic gradient descent optimizer.
    optimizer = tf.optimizers.SGD(par.learning_rate)
    
    start_time = time.time()
    title="Iter     Cost  Accuracy  Elapsed(s)  Block(s) \n"
    print(title)    
    file1.write(title)    
    Best_accuracy=0.0; Best_cost=1e20; Best_W = np.zeros((par.num_features, par.num_classes))
    # Run training for the given number of steps (Randomly select batch_x (256 x 784) and its corresponding batch_y (256 x 1) for each iteration)
    for iteration, (batch_x, batch_y) in enumerate(train_data.take(par.training_steps), 1):        
        Block_time_start = time.time()        
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:            
            pred = logistic_regression(batch_x,W)
            loss = cross_entropy(pred, batch_y, par.num_classes)
        # Compute gradients.        
        gradients = g.gradient(loss, [W])        
        # Update W following gradients.        
        optimizer.apply_gradients(zip(gradients, [W]))

        end_time = time.time()
        Block_time = end_time - Block_time_start
        elapsed_time = end_time - start_time      

        #
        pred = logistic_regression(batch_x,W)
        cost = cross_entropy(pred, batch_y, par.num_classes)
        accuracy = prediction_accuracy(pred, batch_y)    
        Best_accuracy, Best_cost, Best_W = calculate_best_solution(iteration, cost, accuracy, W, Best_accuracy, Best_cost, Best_W)

        if iteration % par.display_step == 0: 
            results = '%4d %8.3f %9.3f %9.3f %9.3f \n' %(iteration, Best_cost, Best_accuracy, elapsed_time, Block_time)  
            print(results)          
            file1.write(results)
        

    # Test model on validation set.
    # pred = logistic_regression(x_test,W,b)
    return Best_W, Best_cost, file1
 
######################################################################################################
###### Algorithms
######################################################################################################
def Centralized_GD(par, x_train, y_train, file1):

    par.W_val = np.ones((par.num_features, par.num_classes)) ## J X K matrix    
    
    start_time = time.time()
    title="Iter     Cost  Accuracy  Elapsed(s)  Block(s) \n"
    print(title)    
    file1.write(title)
    Best_accuracy=0.0; Best_cost=1e20; Best_W = np.zeros((par.num_features, par.num_classes))
    batch_x = tf.convert_to_tensor(x_train)
    batch_y = tf.convert_to_tensor(y_train)
    for iteration in range(1, par.training_steps + 1):        
        Block_time_start = time.time()
        
        W = tf.Variable( tf.convert_to_tensor(par.W_val, np.float32), name="weight"  )        
        
        
        with tf.GradientTape() as g:            
            pred = logistic_regression(batch_x,W)                        
            loss = cross_entropy(pred, batch_y, par.num_classes)

        # Compute gradients.        
        gradients = g.gradient(loss, [W])
        
        ## update        
        par.W_val =  W.numpy() - par.learning_rate *  gradients[0].numpy()
        
        end_time = time.time()
        Block_time = end_time - Block_time_start
        elapsed_time = end_time - start_time  

        #
        pred = logistic_regression(batch_x,W)
        cost = cross_entropy(pred, batch_y, par.num_classes)
        accuracy = prediction_accuracy(pred, batch_y)    
        Best_accuracy, Best_cost, Best_W = calculate_best_solution(iteration, cost, accuracy, W, Best_accuracy, Best_cost, Best_W)

        if iteration % par.display_step == 0: 
            results = '%4d %8.3f %9.3f %9.3f %9.3f \n' %(iteration, Best_cost, Best_accuracy, elapsed_time, Block_time)  
            print(results)          
            file1.write(results)
     
    
    return Best_W, Best_cost, file1


def DP_IADMM(par, x_train_agent, y_train_agent, x_train_new, y_train_new, Algorithm, file1):

    ## Initialization        

    par.W_val = np.zeros((par.num_features, par.num_classes))
    par.Z_val = np.zeros((par.split_number, par.num_features, par.num_classes)); 
    par.Lambdas_val = np.zeros((par.split_number, par.num_features, par.num_classes));
    
    start_time = time.time()    
    title="Iter       Cost  Accuracy  Residual  Elapsed(s)  Block_1(s)  Solve_1(s)  Block_2(s)  Solve_2(s)  Noise_absmean \n"
    print(title)    
    file1.write(title)
    batch_x = tf.convert_to_tensor(x_train_new)
    batch_y = tf.convert_to_tensor(y_train_new)
    
    for iteration in range(1, par.training_steps + 1):     
        
        First_block_time_start = time.time()
        # par, Runtime_1 = First_Block_Problem(par, m1, w) ## Models.py   
        par, Runtime_1 = First_Block_Problem_ClosedForm(par) ## Models.py   
        First_block_time_end = time.time()
        First_block_elapsed_time = First_block_time_end - First_block_time_start

        Second_block_time_start = time.time()
        if Algorithm =="DP_IADMM_Proximal":            
            par.eta = par.beta / math.sqrt(iteration)                 
            # par, Runtime_2 = Proximal_Second_Block_Problem(par, m2, z, x_train_agent, y_train_agent, iteration) ## Models.py
            par, Runtime_2, tilde_xi_mean = Proximal_Second_Block_Problem_ClosedForm(par, x_train_agent, y_train_agent, iteration) ## Models.py

        if Algorithm =="DP_IADMM_Trust":                         
            par.eta = par.beta / iteration*iteration              
            # print("par.eta=", par.eta)
            # par, Runtime_2 = Trust_Second_Block_Problem(par, m2, z, x_train_agent, y_train_agent, iteration) ## Models.py
            par, Runtime_2, tilde_xi_mean = Trust_Second_Block_Problem_ClosedForm(par, x_train_agent, y_train_agent, iteration) ## Models.py
        Second_block_time_end = time.time()
        Second_block_elapsed_time = Second_block_time_end - Second_block_time_start
 
        for p in range(par.split_number):            
            par.Lambdas_val[p] += par.rho*(par.W_val - par.Z_val[p])
         
        end_time = time.time()
        elapsed_time = end_time - start_time      

        # Tensor Flow
        W = tf.Variable( tf.convert_to_tensor(par.W_val, np.float32), name="weight" )                
        pred = logistic_regression(batch_x,W)
        cost = cross_entropy(pred, batch_y, par.num_classes)
        accuracy = prediction_accuracy(pred, batch_y)
        
 
        if iteration % par.display_step == 0: 
            residual = calculate_residual(par)
            
            results = '%4d %10.3f %9.3f %9.3f %9.3f %10.3f %10.3f %10.3f %10.3f %10.6f \n' %(iteration, cost, accuracy, residual, elapsed_time, First_block_elapsed_time, Runtime_1, Second_block_elapsed_time, Runtime_2, tilde_xi_mean)  
            print(results)          
            file1.write(results)            

            
    return par.W_val, cost.numpy(), file1


def DP_IADMM_Proximal_Huang(par, x_train_agent, y_train_agent, x_train_new, y_train_new, Algorithm, file1):

    ## Initialization        
    
    par.W_val = np.zeros((par.num_features, par.num_classes))
    par.Z_val = np.zeros((par.split_number, par.num_features, par.num_classes)); 
    par.Lambdas_val = np.zeros((par.split_number, par.num_features, par.num_classes));
    
    start_time = time.time()    
    title="Iter       Cost  Accuracy  Residual  Elapsed(s)  Block_1(s)  Solve_1(s)  Block_2(s)  Solve_2(s)  Noise_absmean \n"
    print(title)    
    file1.write(title)

    batch_x = tf.convert_to_tensor(x_train_new)
    batch_y = tf.convert_to_tensor(y_train_new)
    
    for iteration in range(1, par.training_steps + 1):        

        First_block_time_start = time.time()                
        
        par, Runtime_1, tilde_xi_mean = Huang_First_Block_Problem_ClosedForm(par, x_train_agent, y_train_agent, iteration) ## Models.py
        First_block_time_end = time.time()
        First_block_elapsed_time = First_block_time_end - First_block_time_start
             
        Second_block_time_start = time.time()        
        par, Runtime_2 = Huang_Second_Block_Problem_ClosedForm(par) ## Models.py           
        Second_block_time_end = time.time()
        Second_block_elapsed_time = Second_block_time_end - Second_block_time_start

        for p in range(par.split_number):            
            par.Lambdas_val[p] += par.rho*(par.W_val - par.Z_val[p])
         
        end_time = time.time()
        elapsed_time = end_time - start_time      

        # Tensor Flow
        W = tf.Variable( tf.convert_to_tensor(par.W_val, np.float32), name="weight"  )                
        pred = logistic_regression(batch_x,W)
        cost = cross_entropy(pred, batch_y, par.num_classes)
        accuracy = prediction_accuracy(pred, batch_y)
        
        if iteration % par.display_step == 0: 
            residual = calculate_residual(par)
            
            results = '%4d %10.3f %9.3f %9.3f %9.3f %10.3f %10.3f %10.3f %10.3f %10.6f \n' %(iteration, cost, accuracy, residual, elapsed_time, First_block_elapsed_time, Runtime_1, Second_block_elapsed_time, Runtime_2, tilde_xi_mean)  
            print(results)          
            file1.write(results)            

            
    return par.W_val, cost.numpy(), file1










# def Distributed_IADMM_Proximal(par, x_train_agent, y_train_agent, x_train_new, y_train_new, file1):

#     ## Initialization        
#     m1, m2, w, z  = Create_Models(par) ## Models.py

#     par.W_val = np.zeros((par.num_features, par.num_classes))
#     par.Z_val = np.zeros((par.split_number, par.num_features, par.num_classes)); 
#     par.Lambdas_val = np.zeros((par.split_number, par.num_features, par.num_classes));
    
#     start_time = time.time()    
#     title="Iter     Cost  Accuracy  Residual  Elapsed(s)  Block_1(s)  Block_2(s) \n"
#     print(title)    
#     file1.write(title)
    
#     for iteration in range(1, par.training_steps + 1):        
        
#         First_block_time_start = time.time()
#         par = First_Block_Problem(par, m1, w) ## Models.py   
#         First_block_time_end = time.time()
#         First_block_elapsed_time = First_block_time_end - First_block_time_start

#         ############################################################
#         par.eta = 1.0 / math.sqrt(iteration)         
#         ############################################################
#         Second_block_time_start = time.time()
#         par = Proximal_Second_Block_Problem(par, m2, z, x_train_agent, y_train_agent, iteration) ## Models.py
#         Second_block_time_end = time.time()
#         Second_block_elapsed_time = Second_block_time_end - Second_block_time_start

#         for p in range(par.split_number):
#             for j in range(par.num_features):    
#                 for k in range(par.num_classes):
#                     par.Lambdas_val[p][j][k] += par.rho*(par.W_val[j][k] - par.Z_val[p][j][k])
                

#         ## Print
#         end_time = time.time()
#         elapsed_time = end_time - start_time        
#         if iteration % par.display_step == 0: 
             
#             H = calculate_hypothesis(par.W_val, x_train_new) ## I x K matrix             
#             ## one-hot vector
#             num_data = y_train_new.shape[0]        
#             y_train_Bin = np.zeros((num_data, par.num_classes)) 
#             for idx in range(num_data):
#                 y_train_Bin[idx][y_train_new[idx]] = 1.
#             cost = calculate_cost(par, num_data, H, y_train_Bin)
#             accuracy = calculate_accuracy(par, par.W_val, x_train_new, y_train_new)
#             residual = calculate_residual(par)
            
#             results = '%4d %8.3f %9.3f %9.3f %9.3f %9.3f %9.3f \n' %(iteration, cost, accuracy, residual, elapsed_time, First_block_elapsed_time, Second_block_elapsed_time)  
#             print(results)          
#             file1.write(results)            

            
#     return par.W_val, cost, file1

# # def Distributed_IADMM_Trust(par, x_train_agent, y_train_agent, x_train_new, y_train_new, file1):

#     ## Initialization        
#     m1, m2, w, z  = Create_Models(par) ## Models.py

#     par.W_val = np.zeros((par.num_features, par.num_classes))
#     par.Z_val = np.zeros((par.split_number, par.num_features, par.num_classes)); 
#     par.Lambdas_val = np.zeros((par.split_number, par.num_features, par.num_classes));
    
#     start_time = time.time()    
#     title="Iter     Cost  Accuracy  Residual  Elapsed(s)  Block_1(s)  Block_2(s) \n"
#     print(title)    
#     file1.write(title)
    
#     for iteration in range(1, par.training_steps + 1):        
        
#         First_block_time_start = time.time()
#         par = First_Block_Problem(par, m1, w) ## Models.py   
#         First_block_time_end = time.time()
#         First_block_elapsed_time = First_block_time_end - First_block_time_start

#         ############################################################
#         par.eta = 1.0 / iteration*iteration
#         ############################################################
#         Second_block_time_start = time.time()
#         par = Proximal_Second_Block_Problem(par, m2, z, x_train_agent, y_train_agent, iteration) ## Models.py
#         Second_block_time_end = time.time()
#         Second_block_elapsed_time = Second_block_time_end - Second_block_time_start

#         for p in range(par.split_number):
#             for j in range(par.num_features):    
#                 for k in range(par.num_classes):
#                     par.Lambdas_val[p][j][k] += par.rho*(par.W_val[j][k] - par.Z_val[p][j][k])
                

#         ## Print
#         end_time = time.time()
#         elapsed_time = end_time - start_time        
#         if iteration % par.display_step == 0: 
            
#             H = calculate_hypothesis(par.W_val, x_train_new) ## I x K matrix             
#             ## one-hot vector
#             num_data = y_train_new.shape[0]        
#             y_train_Bin = np.zeros((num_data, par.num_classes)) 
#             for idx in range(num_data):
#                 y_train_Bin[idx][y_train_new[idx]] = 1.
#             cost = calculate_cost(par, num_data, H, y_train_Bin)
#             accuracy = calculate_accuracy(par, par.W_val, x_train_new, y_train_new)
#             residual = calculate_residual(par)
            
#             results = '%4d %8.3f %9.3f %9.3f %9.3f %9.3f %9.3f \n' %(iteration, cost, accuracy, residual, elapsed_time, First_block_elapsed_time, Second_block_elapsed_time)  
#             print(results)          
#             file1.write(results)            

            
#     return par.W_val, cost, file1