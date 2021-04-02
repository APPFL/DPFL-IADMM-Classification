import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

import math
import csv
import json


def Read_MNIST(num_features):
  ################################################################################################################################################
  ##### From TensorFlow:  MNIST  
  ##### https://builtin.com/data-science/guide-logistic-regression-tensorflow-20  
  ################################################################################################################################################
  # Load MNIST
  (x_train, y_train), (x_test, y_test) = mnist.load_data()  
  
  # Convert to float32.
  x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)  

  # Flatten images to 1-D vector of 784 features (28*28).  
  x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
  
  # Normalize images value from [0, 255] to [0, 1].
  x_train, x_test = x_train / 255., x_test / 255.

  # print("x_train=", type(x_train), len(x_train), x_train.shape )
  # print("y_train=", type(y_train), len(y_train), y_train.shape )
  # print("x_test=", type(x_test), len(x_test), x_test.shape )
  # print("y_test=", type(y_test), len(y_test), y_test.shape )
  # stop
  
  return x_train, x_test, y_train, y_test
   

def Read_FEMNIST(num_features):
  ################################################################################################################################################
  ##### FEMNIST (datatype=dict)
  ##### Example: "users": ["f3795_00", "f3608_13"], "num_samples": [149, 162], "user_data": {"f3795_00": {"x": [], ..., []}, "y": [4, ..., 31]},
  ################################################################################################################################################
  
  TS = 36 ## TS <= 36 for FEMNIST dataset
  
  train_data={}  
  temp_x_train=[]; temp_y_train=[]; total_num_train_data=0;
  for testset in range(TS):
    with open('./Inputs/Train_large/all_data_%s_niid_0_keep_0_train_9.json'%(testset)) as f:
      train_data[testset] = json.load(f)    
    for user in train_data[testset]["users"]:      
      total_num_train_data += len(train_data[testset]["user_data"][user]["y"])

      ## x_train      
      for x_elem in train_data[testset]["user_data"][user]["x"]:
        temp_x_train.append( np.array(x_elem) )      
      ## y_train
      for y_elem in train_data[testset]["user_data"][user]["y"]:        
        temp_y_train.append( y_elem )
      
  # print("1. total_num_train_data=",total_num_train_data)
  x_train = np.array(temp_x_train,np.float32)
  y_train = np.array(temp_y_train,np.uint8)

  test_data={}  
  temp_x_test=[]; temp_y_test=[]; total_num_test_data=0;
  for testset in range(TS):
    with open('./Inputs/Test_large/all_data_%s_niid_0_keep_0_test_9.json'%(testset)) as f:
      test_data[testset] = json.load(f)    
    for user in test_data[testset]["users"]:      
      total_num_test_data += len(test_data[testset]["user_data"][user]["y"])

      ## x_train      
      for x_elem in test_data[testset]["user_data"][user]["x"]:        
        temp_x_test.append( np.array(x_elem) )      ##################
        # temp_x_test.append( 1 - np.array(x_elem) )    
      ## y_train
      for y_elem in test_data[testset]["user_data"][user]["y"]:        
        temp_y_test.append( y_elem )
      
  # print("2. total_num_test_data=",total_num_test_data)
  x_test = np.array(temp_x_test,np.float32)
  y_test = np.array(temp_y_test,np.uint8)


  print("x_train=", type(x_train), len(x_train), x_train.shape )
  print("y_train=", type(y_train), len(y_train), y_train.shape )
  print("x_test=", type(x_test), len(x_test), x_test.shape )
  print("y_test=", type(y_test), len(y_test), y_test.shape )
  
  # stop
  return x_train, x_test, y_train, y_test


def Split_MNIST_Data(x_train, y_train, split_number):

  x_train_agent = np.split(x_train, split_number)
  y_train_agent = np.split(y_train, split_number)  
  
  x_list = []; y_list = [];
  for p in range(split_number):
    x_list.append(x_train_agent[p])
    y_list.append(y_train_agent[p])    

  x_train_new = np.concatenate( np.array(x_list) )
  y_train_new = np.concatenate( np.array(y_list) )  
    
  return x_train_new, y_train_new, x_train_agent, y_train_agent

  
  
   