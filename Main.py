from Read import *
from Algorithms import *
from Structure import *
from Generate_Laplacian_Noise import *
import os
#################################################################################################################################################
############################# Classificatin Problem with a multi-class logistic regression loss function (MNIST data) 
#################################################################################################################################################

def main(Algorithm, rho, beta):

  ## Parameters
  par = Parameters()

  par.training_steps = 1000
  par.display_step = 20
  par.split_number = 10  ## {5, 10, 50, 100, 500}
  par.gamma = 1e-6  ## regularizer
  par.bar_epsilon = 1.0  ## {0.01, 0.05, 0.1, 1.0, 10.0 'none'}
  # SGD parameters.
  par.learning_rate = 0.01 ## MNIST: 0.01;   
  par.batch_size = 60000  ## MNIST: 256;  
  ## Parameters for IADMM
  par.beta = beta
  par.rho = rho
  par.LB = -1.0
  par.UB =  1.0  
  
  ########################################################################################################################################
  # MNIST dataset parameters.
  par.num_features = 784 # 28*28
  par.num_classes = 10 # 0 to 9 digits
  # Read MNIST
  x_train, x_test, y_train, y_test = Read_MNIST(par.num_features)  ## type=np.ndarray (x_train: 60000 x 784 (float:0.~1.), y_train: 60000 x 1 (int:0~9), ... )
  # Split MNIST training data
  x_train_new, y_train_new, x_train_agent, y_train_agent = Split_MNIST_Data(x_train, y_train, par.split_number) ## Read.py
  par.total_data = x_train_new.shape[0]  
  ########################################################################################################################################
  

  
  ## Write output
  filename = 'Results_%s_P_%s_Eps_%s'%(Algorithm,par.split_number, par.bar_epsilon)
  file_ext = '.txt'
  Path = './Outputs/%s%s'%(filename, file_ext)
  uniq = 1
  while os.path.exists(Path):
    Path = './Outputs/%s_%d%s'%(filename, uniq, file_ext)
    uniq += 1
  file1 = open(Path,"w")
  
  
  if Algorithm =="SGD_TF":
    # Use tf.data API to shuffle and batch data.
    train_data=tf.data.Dataset.from_tensor_slices((x_train_new,y_train_new))
    train_data=train_data.repeat().shuffle(5000).batch(par.batch_size).prefetch(1)  
    W, cost, file1 = Centralized_SGD_TensorFlow(par, train_data, file1)


  if Algorithm =="GD":
    W, cost, file1 = Centralized_GD(par, x_train_new, y_train_new, file1)


  if Algorithm =="DP_IADMM_Proximal":
    W, cost, file1 = DP_IADMM(par, x_train_agent, y_train_agent, x_train_new, y_train_new, Algorithm, file1)
    

  if Algorithm =="DP_IADMM_Trust":
    W, cost, file1 = DP_IADMM(par, x_train_agent, y_train_agent, x_train_new, y_train_new, Algorithm, file1)
    # W, cost, file1 = Distributed_IADMM_Trust(par, x_train_agent, y_train_agent, x_train_new, y_train_new, file1)

  if Algorithm == "DP_IADMM_Proximal_Huang":
    W, cost, file1 = DP_IADMM_Proximal_Huang(par, x_train_agent, y_train_agent, x_train_new, y_train_new, Algorithm, file1)

  ## Testing     
  pred = logistic_regression(x_test,W)
  accuracy = prediction_accuracy(pred, y_test)

  #### PRINT & WRITE
  # Laplace_shape = "Laplace_shape=%s \n"%(par.bar_lambda)
  rho_value = "%s_rho=%s \n"%(Algorithm, par.rho )
  beta_value = "%s_beta=%s \n"%(Algorithm, par.beta )
  training_cost = "%s_cost (training)=%s \n"%(Algorithm, cost )
  testing_accuracy = "%s_accuracy (testing)=%s \n"%(Algorithm, accuracy.numpy() )
  # print(Laplace_shape)
  print(rho_value)
  print(beta_value)
  print(training_cost)
  print(testing_accuracy)
  file1.write("\n \n")
  # file1.write(Laplace_shape)
  file1.write(rho_value)
  file1.write(beta_value)
  file1.write(training_cost)
  file1.write(testing_accuracy)
  file1.close()
  
 
###############################################
# Algorithm = "DP_IADMM_Proximal" ## SGD_TF, GD, DP_IADMM_Proximal, DP_IADMM_Trust

main("DP_IADMM_Trust", 2.5, 1)
# main("DP_IADMM_Proximal", 2.5, 1)
# main("DP_IADMM_Proximal_Huang", 2.5, 1)


# for rho in [2.5, 10]:
#   for algo in ["DP_IADMM_Proximal", "DP_IADMM_Trust", "DP_IADMM_Proximal_Huang"]: 
#       main(algo, rho, 1.0)
  