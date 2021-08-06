# Differentially Private Inexact ADMM for a Federated Learning model

In this open-source code, we implement inexact differentially private alternating direction method of multipliers (DP-IADMM) algorithms for solving a distributed empirical risk minimization problem with the multi-class logistic regression function.
In specific, the following three algorithms are implemented:

- "ObjT":  DP-IADMM with a trust-region of solutions incorporated with the objective perturbation method. 
- "ObjP":  DP-IADMM with a proximal function incorporated with the objective perturbation method.
- "OutP": DP-IADMM with a proximal function incorporated with the output perturbation method.
 
## Install and Run 

```
git clone https://github.com/minseok-ryu/DP-IADMM-Multiclass-Logistic.git
```

After downloading the code, open the terminal and go to the directory where "run_1.py" exists.

1. Do the followings:

If you have NVIDIA GPU:

```
conda env create -f environment_gpu.yml
pip install -r requirements_gpu.txt
```	

Otherwise, 

```
conda env create -f environment_cpu.yml
pip install -r requirements_cpu.txt
```	

2. Run:

```
python run_1.py
```	

3. Go to "Outputs" directory to see the results 

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
