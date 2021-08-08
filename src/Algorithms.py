import math
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# TODO: deriving sub-classes for different algorithms
class DP_IADMM_torch:
    def __init__(self, par, x_train_agent, y_train_agent, x_test, y_test, file1):

        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on", self.device)

        self.par = par
        self.x_train_agent = x_train_agent
        self.y_train_agent = y_train_agent
        self.x_test = x_test
        self.y_test = y_test
        self.file1 = file1

        self.num_data = []
        self.x_train_sum = []
        self.y_train_Bin = []
        for p in range(par.split_number):
            self.num_data.append(x_train_agent[p].shape[1])
            self.x_train_sum.append(torch.sum(x_train_agent[p], 1).to(self.device))
            self.y_train_Bin.append(
                (torch.arange(par.num_classes, dtype=torch.int8, device=self.device) == y_train_agent[p][..., None])
                .type(torch.int8)
            )

        # Variables
        # Global model parameters
        self.W_val = torch.zeros(
            par.num_features, par.num_classes, dtype=torch.float32, device=self.device
        )
        # Local model parameters defined for every agent
        self.Z_val = torch.zeros(
            par.split_number,
            par.num_features,
            par.num_classes,
            dtype=torch.float32,
            device=self.device,
        )
        self.Z_next = torch.zeros(
            par.split_number,
            par.num_features,
            par.num_classes,
            dtype=torch.float32,
            device=self.device,
        )
        self.Lambdas_val = torch.zeros(
            par.split_number,
            par.num_features,
            par.num_classes,
            dtype=torch.float32,
            device=self.device,
        )  # Dual variables

        # Matrix Normal Distribution used for Gaussian Mechanism for "Base" algorithm
        self.M = torch.zeros(
            par.num_features, par.num_classes, dtype=torch.float32, device=self.device
        )
        self.U = torch.zeros(
            par.num_features, par.num_features, dtype=torch.float32, device=self.device
        )
        self.V = torch.zeros(
            par.num_classes, par.num_classes, dtype=torch.float32, device=self.device
        )
        self.tilde_xi = torch.zeros(
            par.num_features, par.num_classes, dtype=torch.float32, device=self.device
        )

        self.linear = torch.nn.ModuleList([torch.nn.Linear(par.num_features, par.num_classes, bias=False).to(self.device) for _ in range(par.split_number)])
        self.loss_value = []
        for p in range(par.split_number):
            self.linear[p].weight.data.fill_(0.0)
            self.loss_value.append(
                torch.empty(self.num_data[p], dtype=torch.float16, device=self.device)
            )

        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

        self.Avg_Noise_Mag = 0
        self.Grad_Time = 0.0
        self.Noise_Time = 0.0
        self.update_z_time = 0.0

    def solve(self):

        start_time = time.time()
        title = "Iter    TrainCost     TestAcc     Violation    Elapsed(s)   Solve_1(s)   Solve_2(s)    GradT(s)    NoiseT(s)  AbsNoiseMag    Z_change     AdapRho \n"
        title = (
            "%12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s \n"
            % (
                "Iter",
                "TrainCost",
                "TestAcc",
                "Violation",
                "Elapsed(s)",
                "Solve_1(s)",
                "Solve_2(s)",
                "GradT(s)",
                "NoiseT(s)",
                "UpdateT(s)",
                "AbsNoiseMag",
                "Z_change",
                "AdapRho",
            )
        )
        print(title, end="")
        self.file1.write(title)

        Runtime_1 = 0.0
        Runtime_2 = 0.0
        self.Grad_Time = 0.0
        self.Noise_Time = 0.0

        for iteration in range(self.par.training_steps + 1):

            # Hyperparameter Rho
            self.hyperparameter_rho(iteration)

            stime = time.time()

            # [1] First Block Problem
            if self.par.Algorithm == "OutP":
                self.Base_First_Block_Problem_ClosedForm(iteration)
            else:
                self.First_Block_Problem_ClosedForm()

            Runtime_1 += time.time() - stime

            stime = time.time()

            # [2] Second Block Problem
            if self.par.Algorithm == "OutP":
                self.Base_Second_Block_Problem_ClosedForm()
            else:
                # with profile(
                #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                #     record_shapes=True,
                #     with_stack=True,
                #     profile_memory=False,
                # ) as prof:
                #     with record_function("Second_Block_Problem_ClosedForm"):
                #         self.Second_Block_Problem_ClosedForm(iteration)
                # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                self.Second_Block_Problem_ClosedForm(iteration)

            Runtime_2 += time.time() - stime

            # [3] Dual update
            self.Lambdas_val += self.par.rho * (self.W_val - self.Z_val)

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Display intermediat results
            if iteration % self.par.display_step == 0:

                # Compute cost testing accuracy
                cost = self.calculate_cost()
                accuracy_test = self.calculate_accuracy()

                # Compute concensus violation
                residual = (
                    torch.sum(torch.linalg.norm(self.W_val - self.Z_val, 1, 0))
                    / self.par.split_number
                )

                results = (
                    "%12d %12.6e %12.6e %12.6e %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12.6e %12.6e %12.6e \n"
                    % (
                        iteration,
                        cost,
                        accuracy_test,
                        residual,
                        elapsed_time,
                        Runtime_1,
                        Runtime_2,
                        self.Grad_Time,
                        self.Noise_Time,
                        self.update_z_time,
                        self.Avg_Noise_Mag,
                        self.Z_Change.mean(),
                        self.par.rho,
                    )
                )
                print(results, end="")
                self.file1.write(results)

        # Compute cost and testing accuracy
        cost = self.calculate_cost()
        accuracy_test = self.calculate_accuracy()

        return float(cost), float(accuracy_test)

    def hyperparameter_rho(self, iteration):
        par = self.par
        if par.rho_str == "dynamic_1" or par.rho_str == "dynamic_2":
            if par.Instance == "MNIST":
                c1 = 2.0
                c2 = 5.0
                Tc = 10000.0
                rhoC = 1.2
            elif par.Instance == "FEMNIST":
                c1 = 0.005
                c2 = 0.05
                Tc = 2000.0
                rhoC = 1.2
            elif par.Instance == "CIFAR10":
                c1 = 2.0
                c2 = 5.0
                Tc = 10000.0
                rhoC = 1.2
            else:
                raise AssertionError(
                    "Unexpected value of 'par.Instance'!", par.Instance
                )

            if par.bar_eps_str == "infty":
                par.rho = c1 * math.pow(rhoC, math.floor((iteration + 1) / Tc))
            else:
                par.rho = c1 * math.pow(
                    rhoC, math.floor((iteration + 1) / Tc)
                ) + c2 / float(par.bar_eps_str)
            if par.rho_str == "dynamic_2":
                par.rho = par.rho / 100.0
        else:
            par.rho = float(par.rho_str)

        # the parameter is bounded above
        if par.rho > 1e9:
            par.rho = 1e9

    def Base_First_Block_Problem_ClosedForm(self, iteration):

        self.Z_Change = 0
        self.Avg_Noise_Mag = 0
        for p in range(self.par.split_number):

            stime = time.time()

            output = self.linear[p](self.x_train_agent[p])
            self.loss_value[p] = self.loss(output, self.y_train_agent[p].long())
            self.loss_value[p].mean().backward()
            grad = self.linear[p].weight.grad.t()

            self.Grad_Time += time.time() - stime

            # Decide eta
            self.calculate_eta_Base(self.num_data[p], iteration)

            # Update Z_val
            self.Z_next[p] = (1.0 / (self.par.rho + (1.0 / self.par.eta))) * (
                -grad
                + self.par.rho * self.W_val
                + self.Lambdas_val[p]
                + (1.0 / self.par.eta) * self.Z_val[p]
            )

            self.Z_Change += torch.absolute(self.Z_val[p] - self.Z_next[p])
            self.Z_val[p] = self.Z_next[p]

            # Generate Matrix Normal Noise
            if self.par.bar_eps_str != "infty":

                stime = time.time()
                self.generate_matrix_normal_noise(self.num_data[p])
                self.Noise_Time += time.time() - stime

                self.Z_val[p] += self.tilde_xi
                self.Avg_Noise_Mag += torch.mean(torch.absolute(self.tilde_xi))

        self.Avg_Noise_Mag = self.Avg_Noise_Mag / self.par.split_number

    def Base_Second_Block_Problem_ClosedForm(self):
        self.W_val = (
            torch.sum(self.Z_val - self.Lambdas_val / self.par.rho, axis=0)
            / self.par.split_number
        )

    def First_Block_Problem_ClosedForm(self):
        self.W_val = (
            torch.sum(self.Z_val - self.Lambdas_val / self.par.rho, axis=0)
            / self.par.split_number
        )

    def Second_Block_Problem_ClosedForm(self, iteration):

        self.Z_Change = 0
        self.Avg_Noise_Mag = 0
        objt_time = 0.0
        for p in range(self.par.split_number):

            stime = time.time()

            output = self.linear[p](self.x_train_agent[p])
            self.loss_value[p] = self.loss(output, self.y_train_agent[p].long())
            self.loss_value[p].mean().backward()
            grad = self.linear[p].weight.grad.t()

            self.Grad_Time += time.time() - stime

            # Generate Laplacian Noise
            if self.par.bar_eps_str != "infty":
                stime = time.time()
                self.generate_laplacian_noise(p, output)
                self.Avg_Noise_Mag += torch.mean(torch.absolute(self.tilde_xi))
                self.Noise_Time += time.time() - stime

            # Update Z_val
            if self.par.Algorithm == "ObjP":
                self.par.eta = float(self.par.a_str) / math.sqrt(iteration + 1)
                Z_Prev = self.Z_val[p]
                self.Z_next[p] = (1.0 / (self.par.rho + (1.0 / self.par.eta))) * (
                    -grad
                    + self.par.rho * self.W_val
                    + self.Lambdas_val[p]
                    - self.tilde_xi
                    + (1.0 / self.par.eta) * Z_Prev
                )
                self.Z_Change += torch.absolute(Z_Prev - self.Z_next[p])
                self.Z_val[p] = self.Z_next[p]

            elif self.par.Algorithm == "ObjT":
                stime = time.time()

                self.par.eta = float(self.par.a_str) / (iteration + 1) * (iteration + 1)
                Z_Prev = self.Z_val[p]
                self.Z_next[p] = (1.0 / self.par.rho) * (
                    -grad
                    + self.par.rho * self.W_val
                    + self.Lambdas_val[p]
                    - self.tilde_xi
                )
                # Trust-Region using the infinity norm
                self.Z_next[p] = torch.max(
                    torch.min(self.Z_next[p], Z_Prev + self.par.eta), Z_Prev - self.par.eta
                )
                self.Z_Change += torch.absolute(Z_Prev - self.Z_next[p])
                self.Z_val[p] = self.Z_next[p]

                self.update_z_time += time.time() - stime

        self.Avg_Noise_Mag = self.Avg_Noise_Mag / self.par.split_number

    def calculate_eta_Base(self, num_data, iteration):

        delta = 1e-6  # (epsilon, delta)-differential privacy
        c1 = num_data * 1
        c3 = num_data * 0.25
        cw = math.sqrt(self.par.num_features * self.par.num_classes * 4)

        if self.par.bar_eps_str != "infty":
            self.par.eta = 1.0 / (
                c3
                + 4.0
                * c1
                * math.sqrt(
                    self.par.num_features
                    * self.par.num_classes
                    * (iteration + 1)
                    * math.log(1.25 / delta)
                )
                / (num_data * float(self.par.bar_eps_str) * cw)
            )
        else:
            self.par.eta = 1.0 / c3

        self.par.eta = self.par.eta * float(self.par.a_str)

    def generate_matrix_normal_noise(self, num_data):
        c1 = num_data * 1
        delta = 1e-6  # 1e-308, 1e-6

        sigma = (
            2
            * c1
            * math.sqrt(2 * math.log(1.25 / delta))
            / (
                num_data
                * float(self.par.bar_eps_str)
                * (self.par.rho + 1.0 / self.par.eta)
            )
        )

        tilde_xi_shape = self.M + sigma * sigma
        self.tilde_xi = torch.empty(
            self.par.num_features, self.par.num_classes
        ).normal_(self.M, tilde_xi_shape)

    def generate_laplacian_noise(self, p, output):

        H_hat_abs_sum = torch.linalg.norm(
            self.softmax(output) - self.y_train_Bin[p], 1, 1
        )
        x_train_H_hat_abs = (
            torch.multiply(self.x_train_sum[p], H_hat_abs_sum) / self.par.total_data
        )
        bar_lambda = torch.max(x_train_H_hat_abs) / float(self.par.bar_eps_str)

        tilde_xi_shape = self.M + bar_lambda
        m = torch.distributions.laplace.Laplace(self.M, tilde_xi_shape)
        self.tilde_xi = m.sample()

    def calculate_accuracy(self):
        accuracy_test = 0.0
        test_output = torch.argmax(self.linear[1](self.x_test), 1)
        for p in range(self.par.split_number):
            test_output = torch.argmax(self.linear[p](self.x_test), 1)
            accuracy_test += (
                (test_output == self.y_test).type(dtype=torch.float16).mean()
            )

        return accuracy_test / self.par.split_number

    def calculate_cost(self):
        cost = 0.0
        for p in range(self.par.split_number):
            cost += self.loss_value[p].mean()

        return cost / self.par.split_number

