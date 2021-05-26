# Differentially Private Inexact ADMM for a Federated Learning model

In this open-source library, we implement inexact alternating direction method of multipliers (IADMM) algorithms for solving a distributed empirical risk minimization problem with the multi-class logistic regression function.
In specific, the following three algorithms are implemented:
-"OutP":
-"ObjP":
-"ObjT":
 
 
 
# Install and Run 
After downloading the code, open the terminal and go to the directory where "run_1.py" exists.
[1] Do the followings:
	conda create -n DPFL
	conda activate DPFL
	conda install numpy
	conda install cupy (for GPU computation)
	pip install GPUInfo (for GPU computation)
	pip install mlxtend (for MNIST dataset)
[2] Run:
	python run_1.py
[3] Go to "Outputs" directory to see the results 

# Important Note
To use GPU, change from "import numpy as np" to  "import cupy as np" located in the first line of Algorithms.py, Models.py, Read.py, and Functions.py.