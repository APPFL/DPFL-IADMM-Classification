from Read import *
from Algorithms import *
from Structure import *
import os
from gpuinfo import GPUInfo

#################################################################################################################################################
## Differentially Private Inexact ADMM for solving a distributed ERM with a multiclass logistic regression loss function 
#################################################################################################################################################

def main(Instance, Agent, Algorithm, Hyperparameter, ScalingConst, Epsilon, TrainingSteps, DisplaySteps):
  
  GPU = GPUInfo.check_empty() # Check if GPU is used.
  if len(GPU) > 0:
    print("Running on GPU:", GPU)
  else:
    print("Running on CPU:", GPU)

  ## Parameters
  par = Parameters()
  par.Instance = Instance
  par.Algorithm = Algorithm
  par.rho_str = Hyperparameter
  par.a_str = ScalingConst    
  par.bar_eps_str = Epsilon   
  par.training_steps = int(TrainingSteps)
  par.display_step = int(DisplaySteps)
  par.gamma = 1e-6  ## regularizer parameter  
  
  ## Read Instance
  if par.Instance =="MNIST":            
    x_test, y_test, x_train_new, y_train_new, x_train_agent, y_train_agent = Read_MNIST(par, Agent)   
    par.total_data = x_train_new.shape[0]  
  elif par.Instance =="FEMNIST":        
    x_test, y_test, x_train_new, y_train_new, x_train_agent, y_train_agent = Read_FEMNIST(par, Agent)
    par.total_data = x_train_new.shape[0]    
  elif par.Instance =="CIFAR10":        
    x_test, y_test, x_train_new, y_train_new, x_train_agent, y_train_agent = Read_CIFAR10(par, Agent)
    par.total_data = x_train_new.shape[0]
  else:
    raise AssertionError("Unexpected value of 'par.Instance'!", par.Instance)
  # print("par.total_data=",par.total_data)  
  
  #### Write output  
  foldername = "Outputs"  
  filename = 'Results_%s_Agent_%s_%s_rho_%s_a_%s_eps_%s'%(par.Instance, par.split_number, par.Algorithm, par.rho_str, par.a_str, par.bar_eps_str)
  
  file_ext = '.txt'
  Path = './%s/%s%s'%(foldername,filename, file_ext)
  uniq = 1
  while os.path.exists(Path):
    Path = './%s/%s_%d%s'%(foldername,filename, uniq, file_ext)
    uniq += 1
  file1 = open(Path,"w")

  #### Training Process     
  W, cost, file1 = DP_IADMM(par, x_train_agent, y_train_agent, x_train_new, y_train_new, x_test, y_test, file1)
    
  #### Testing Accuracy
  accuracy = calculate_accuracy(par, W, x_test, y_test)

  #### PRINT & WRITE
  GPU_is = "GPU=%s \n"%(GPU)  
  Instance_Name = "Instance=%s \n"%(par.Instance)
  Agent_num =  "#Agents=%s \n"%(par.split_number)  
  Feature_num =  "#Features=%s \n"%(par.num_features)  
  Class_num =  "#Classes=%s \n"%(par.num_classes)  
  Algorithm_Name = "Algorithm=%s \n"%(par.Algorithm)  
  Hyperparameter_Name = "Hyperparameter_rho=%s \n"%( par.rho_str )
  ScalingConst_Name = "ScalingConst_a=%s \n"%( par.a_str )  
  DP_Epsilon = "DP_Epsilon=%s \n"%(par.bar_eps_str)  
  training_cost = "cost (training)=%s \n"%( cost )
  testing_accuracy = "accuracy (testing)=%s \n"%( accuracy )
  
  print(GPU_is, end='')
  print(Instance_Name, end='')
  print(Agent_num, end='')    
  print(Feature_num, end='')
  print(Class_num, end='')
  print(Algorithm_Name, end='')
  print(Hyperparameter_Name, end='')
  print(ScalingConst_Name, end='')
  print(DP_Epsilon, end='')
  print(training_cost, end='')
  print(testing_accuracy, end='')
  file1.write("\n \n")
  file1.write(GPU_is)
  file1.write(Instance_Name)  
  file1.write(Agent_num)
  file1.write(Feature_num)
  file1.write(Class_num)
  file1.write(Algorithm_Name)  
  file1.write(Hyperparameter_Name)
  file1.write(ScalingConst_Name)  
  file1.write(DP_Epsilon)  
  file1.write(training_cost)
  file1.write(testing_accuracy)
  file1.close() 
