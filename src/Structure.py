import csv

class Parameters:
    def _init_(self):
        self.Instance = ""
        self.Algorithm = ""
        self.rho_str = ""
        self.a_str = ""
        self.bar_eps_str = ""

        self.total_data = 0
        self.parameter_size = 0
        self.split_number = 0
        self.gamma = 0
        self.rho_const = 0
        self.rho = 0
        self.beta_const = 0
        self.beta = 0
        self.eta = 0
        self.UB = 0
        self.LB = 0
        self.bar_epsilon = 0
        self.Iteration_Limit = 0
        self.print_iter = 0
        self.tilde_xi = []
        self.bar_lambda = 0
        
        self.W_val = [] 
        self.Z_val = [] 
        self.Lambdas_val = [] 
        self.M=[]
        self.U=[]
        self.V=[]


        ####
        self.learning_rate = 0
        self.training_steps = 0
        self.batch_size = 0
        self.display_step = 0        
        self.num_features = 0
        self.num_classes = 0
       
        
        

            

    
