import numpy as np
import tensorflow as tf
import math
from scipy.stats import matrix_normal

def calculate_best_solution(iteration, cost, accuracy, W, Best_accuracy, Best_cost, Best_W):
    
    if iteration == 1:
        Best_accuracy = accuracy.numpy()
        Best_cost = cost.numpy()
        Best_W = W.numpy()
    else:
        if cost.numpy() < Best_cost:
            Best_accuracy = accuracy.numpy()
            Best_cost = cost.numpy()
            Best_W = W.numpy()
    
    return Best_accuracy, Best_cost, Best_W


def logistic_regression(x,W):    
    # Apply softmax to normalize the logits to a probability distribution.    
    return tf.nn.softmax(tf.matmul(x, W))
 
def cross_entropy(y_pred, y_true, num_classes):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

def prediction_accuracy(y_pred, y_true):
  # Predicted class is the index of the highest score in prediction vector (i.e. argmax).
  correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
  return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



def calculate_hypothesis(W_val, x_train):
    
    Temp_H = np.exp( np.matmul(x_train, W_val) )
    # Temp_H = np.clip(Temp_H,1e-9, 1e9) ## I X K matrix
    H=[]
    for row in Temp_H:        
        H.append(row/np.sum(row))
    H = np.array(H)   
    H = np.clip(H,1e-9, 1.) ## I X K matrix
 
    return H

def calculate_cost(par, num_data, H, y_train_Bin):
    return -np.sum( np.multiply(y_train_Bin, np.log(H)) ) / num_data + par.gamma * np.sum( np.square(par.W_val) )

def calculate_accuracy(par, W, x_train, y_train):
    H = calculate_hypothesis(W, x_train)
    H_argmax = np.argmax(H, axis=1)        
    return np.mean( np.equal(H_argmax, y_train ) )

def calculate_gradient(par, num_data, H, x_train, y_train_Bin):
    grad = np.zeros((par.num_features, par.num_classes))    ## J X K matrix
    H_hat=np.subtract(H, y_train_Bin)    ## I X K matrix
    grad = np.matmul(x_train.transpose(), H_hat) / num_data + 2. * par.gamma * par.W_val ## J X K matrix
    
    return grad

def generate_laplacian_noise(par, H, num_data, x_train, y_train_Bin, tilde_xi):
    H_hat=np.subtract(H, y_train_Bin)    ## I_p X K matrix

    bar_Delta = 0.0;
    for i in range(num_data):
        # S = np.outer(x_train[i], H_hat[i]) / num_data  ## J  X K
        S = np.outer(x_train[i], H_hat[i]) / par.total_data  ## J  X K
        S = np.absolute(S)
        Total_sum = np.sum(S)
        if Total_sum > bar_Delta:
            bar_Delta = Total_sum
    bar_Delta = bar_Delta / 2.0
    

    # np.matmul(x_train.transpose(), H_hat) / num_data

    # #### Sensitivity (Multi-class Logistic Regression)    
    # bar_Delta = round(par.num_features*par.num_classes/par.total_data, 4)
    
    #### Laplace Distribution Shape Parameter    
    bar_lambda = 2.0*bar_Delta/par.bar_epsilon
    # print("bar_Delta=", bar_Delta, "  bar_lambda=",bar_lambda)
    # stop
    ####
    
    np.random.seed(10)
    for j in range(par.num_features):
        for k in range(par.num_classes):
            tilde_xi[j][k] = np.random.laplace(0.0, bar_lambda, 1)    

    # print(tilde_xi)
    # stop            
    return tilde_xi

def calculate_eta_Huang(par,num_data, Iteration):
    
    delta = 1e-308  ##1e-308
    c1 = num_data*1
    c3 = num_data*0.25
    cw = math.sqrt( par.num_features*par.num_classes*4 )
    
    par.eta = 1.0 / ( c3 + 4.0*c1*math.sqrt( par.num_features*par.num_classes*Iteration*math.log(1.25/delta)  )/(num_data*par.bar_epsilon*cw)  )
        
    return par.eta
def generate_matrix_normal_noise(par, num_data,tilde_xi):
    c1 = num_data*1
    delta = 1e-308  ## 1e-308, 1e-6

    sigma = 2*c1*math.sqrt(2*math.log(1.25/delta)) / (num_data*par.bar_epsilon*(par.rho + 1.0/par.eta))
    

    
    M = np.zeros((par.num_features, par.num_classes))  
    U = np.zeros((par.num_features, par.num_features))  
    V = np.zeros((par.num_classes, par.num_classes))  
    for j in range(par.num_features):
        U[j][j] = sigma*sigma
    for k in range(par.num_classes):
        V[k][k] = sigma*sigma        
    
    np.random.seed(10)
    tilde_xi = matrix_normal.rvs(M,U,V)
    
    # print(tilde_xi)
    # stp
    return tilde_xi

def calculate_residual(par):
    residual=0
    for p in range(par.split_number):
        for j in range(par.num_features):         
            for k in range(par.num_classes):
                residual += abs( par.W_val[j][k] - par.Z_val[p][j][k] )          
    residual = residual / par.split_number
    return residual
