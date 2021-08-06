import sys
sys.path.append("./src")
from Main import *

# [1] Instances:
# "MNIST": total number of training data => I = 60000; the number of agents => P \in \{5, 10, 50, 100, ... \}
# "FEMNIST":
# (large)  Original FEMNIST data from leaf (https://leaf.cmu.edu)
# (medium) Extract 25% of the original data
# (small)  Extract  5% of the original data
# Note: the number of agents for FEMNIST is given, e.g., P=195 for small FEMNIST.

# [2] Algorithms:
# "OutP":  DP-IADMM-Prox  (output perturbation)
# "ObjP":  DP-IADMM-Prox  (objective perturbation)
# "ObjT": DP-IADMM-Trust (objective perturbation)

# [3] Hyperparameter \rho^t
# "static":  \rho^t \in \{ "0.1", "1.0", "10.0" \}
# "dynamic":  "dynamic_1", "dynamic_2" (defined in "hyperparameter_rho" in "Functions.py")

# [4] Scaling constant a > 0 for the proximity parameters \eta^t=1/\sqrt{t} and \delta^t=1/t^2
## default: a = "1.0"

# [5] Privacy parameter \bar{\epsilon} \in \{"0.01", "0.05", "0.1", "1.0", "10.0", "infty"\}

# [6] Training_step (Total iteration) and Display_step (display intermediate results)
# Training_step = \{ "20000", "500000", "1000000" \}
# Display_step = \{ "200", "5000", "10000" \}

# Example:
# main("MNIST","10", "ObjT", "dynamic_1", "1.0", "0.05", "100", "1")
main("CIFAR10", "10", "ObjT", "dynamic_1", "1000.0", "infty", "10", "1")
# main("CIFAR10","10", "ObjT", "dynamic_1", "1000.0", "infty", "1000000", "10000")


# main("MNIST","10", "OutP", "dynamic_1", "1.0", "0.05", "20000", "200")
# main("MNIST","10", "ObjP", "dynamic_1", "1.0", "0.05", "20000", "200")
# main("MNIST","10", "ObjT", "dynamic_1", "1.0", "0.05", "20000", "200")
# main("FEMNIST","small", "OutP", "dynamic_1", "1.0", "0.05", "20000", "200")
# main("FEMNIST","small", "ObjP", "dynamic_1", "1.0", "0.05", "20000", "200")
# main("FEMNIST","small", "ObjT", "dynamic_1", "1.0", "0.05", "20000", "200")

# for eps in ["0.05", "0.1", "1.0"]:
#     main("MNIST","10", "ObjT", "dynamic_1", "1.0", eps, "20000", "200")
#     main("FEMNIST","small", "ObjT", "dynamic_1", "1.0", eps, "20000", "200")
