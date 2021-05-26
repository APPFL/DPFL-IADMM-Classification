# Differentially Private Inexact ADMM for a Federated Learning model

In this open-source library, we implement inexact differentially private alternating direction method of multipliers (DP-IADMM) algorithms for solving a distributed empirical risk minimization problem with the multi-class logistic regression function.
In specific, the following three algorithms are implemented:

- "OutP": DP-IADMM with a proximal function incorporated with the output perturbation method.
- "ObjP":  DP-IADMM with a proximal function incorporated with the objective perturbation method.
- "ObjT":  DP-IADMM with a trust-region of solutions incorporated with the objective perturbation method. 
 
# Install and Run 

git clone https://github.com/minseok-ryu/DP-IADMM-Multiclass-Logistic.git

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