import numpy as np

class Player:
    def __init__(self, 
            id: int,
            D_target: list, 
            G_res: list, 
            a: float,
            b: float,
            d: float,
            a_tilde: float, 
            b_tilde: float,
            D_min: float, 
            D_max: float,
            G_min: float,
            G_max: float,
            risk_aversion: float,
            kappa: list, 
            trading_cost: list,
            connections: list,
            probabilities: list,
            ) -> None:

        self.id = id #Simply a serial number of the agent assigned on the first initialization

        self.D_target = D_target
        self.G_res = G_res
        self.a = a
        self.b = b
        self.d = d
        self.a_tilde = a_tilde
        self.b_tilde = b_tilde
        self.D_min = D_min
        self.D_max = D_max
        self.G_min = G_min
        self.G_max = G_max
        self.risk_aversion = risk_aversion
        self.probabilities = probabilities

        self.probabilities_ind = [i for i in range(len(self.probabilities))]


        self.trading_cost = trading_cost
        self.connections = connections

        self.kappa = np.ma.MaskedArray(kappa, 
                                        mask = np.logical_not(self.connections), 
                                        fill_value = 0).filled() 

        self.G = np.zeros(len(self.probabilities_ind))
        self.D = np.zeros(len(self.probabilities_ind))
        self.eta = 0
        self.u = np.zeros(len(self.probabilities_ind))
        self.q = np.zeros((self.community_size, len(self.probabilities_ind)))

        self.grad_G = np.zeros(len(self.probabilities_ind))
        self.grad_D = np.zeros(len(self.probabilities_ind))
        self.grad_eta = 0
        self.grad_u = np.zeros(len(self.probabilities_ind))
        self.grad_q = np.zeros((self.community_size, len(self.probabilities_ind)))

        self.q_others = {}
        self.w_others = {}

        self.plot_d = [[] for i in range(len(self.probabilities))]
        self.plot_g = [[] for i in range(len(self.probabilities))]
        self.plot_u = [[] for i in range(len(self.probabilities))]
        self.plot_eta = []
        self.plot_q = [[[]for i in range(len(self.probabilities))] for i in range(self.community_size)]

    def vector_to_variables(self, x: list) -> None:
        """Method takes as input the concatenated vector of all the variables
        of the agent and assigns the corresponding components to the 
        variables stored in the class instance"""
        
        self.D = x[:len(self.probabilities_ind)]
        self.G = x[len(self.probabilities_ind) : 2*len(self.probabilities_ind)]
        self.eta = x[2*len(self.probabilities_ind)]
        self.u = x[2*len(self.probabilities_ind):3*len(self.probabilities_ind)+1]
        self.q = np.reshape(x[3*len(self.probabilities_ind)+1:],
                        (self.community_size, len(self.probabilities_ind)))

    def variables_to_vector(self) -> np.ndarray:
        """Method concatenates all the variables of the agent and returns a
        vector with the following order of the variables:
        D^1_n,..., D^W_n, G^1_n,..., G^W_n, eta_n, u^1_n,...u^W_n, q^1_{n1}, q^1_{n2}, ..., q^1{nN}, q^2_{n1}, q^2_{n2}, ..., q^W_{nN} """

        return np.concatenate((self.D, 
                            self.G, 
                            np.array([self.eta]), 
                            self.u, 
                            np.reshape(self.q, (self.q.size, ))))

    def utility_generation(self) -> np.ndarray:

        return np.array([0.5*self.a*self.G[proba]**2 + self.b*self.G[proba] + self.d for proba in self.probabilities_ind])


    def utility_demand(self) -> np.ndarray:

        return np.array([self.a_tilde*(self.D[proba] - self.D_target[proba])**2 - self.b_tilde for proba in self.probabilities_ind])


    def q_sum(self, weights: bool = False) -> np.ndarray:
        
        res = np.zeros_like(self.probabilities_ind, dtype= float)

        for proba in self.probabilities_ind:
            actual_trades = np.ma.MaskedArray(self.q.T[proba], 
                                            mask = np.logical_not(self.connections), 
                                            fill_value = 0).filled() 

            if weights:
                res[proba] = np.dot(actual_trades, self.trading_cost)
            else:
                res[proba] = actual_trades.sum()

        return res

    def utility_trading(self) -> np.ndarray:
        return self.q_sum(weights= True)

    def utility(self) -> np.ndarray:

        return self.utility_demand() + self.utility_generation() + self.utility_trading()

    