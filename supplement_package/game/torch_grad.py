from os import stat
import torch
import numpy as np

class TorchPlayer:
    community_size = 3

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
            alpha: list,
            gamma: list,
            insurance_bound: list
            ) -> None:

        self.probabilities = probabilities

        self.probabilities_ind = [i for i in range(len(self.probabilities))]
        
        self.alpha = alpha
        self.gamma = gamma
        self.j_max = insurance_bound

        #self.grad_j = torch.zeros(len(self.probabilities), dtype= float)
        #self.grad_w = torch.zeros(len(self.probabilities), dtype= float)

        self.plot_j = [[] for i in range(len(self.probabilities))]
        self.plot_w = [[] for i in range(len(self.probabilities))] 
        self.plot_u = [[] for i in range(len(self.probabilities))]
        self.plot_eta = [] 

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

        self.w_others = {num: [0 for proba in self.probabilities_ind] for num in range(self.community_size)} #keys: agents' ids, values: agents' ws

        self.q_others = {num: [[0 for proba in self.probabilities_ind] for i in range(self.community_size)] for num in range(self.community_size)}

        self.trading_cost = trading_cost
        self.connections = connections

        self.kappa = np.ma.MaskedArray(kappa, 
                                        mask = np.logical_not(self.connections), 
                                        fill_value = 0).filled()

        self.G = torch.zeros(len(self.probabilities_ind), requires_grad=True)
        self.D = torch.zeros(len(self.probabilities_ind), requires_grad=True)
        self.q = torch.zeros((self.community_size, len(self.probabilities_ind)), requires_grad=True)
        
        self.j = torch.zeros(len(self.probabilities), dtype= float, requires_grad=True)
        self.w = torch.zeros(len(self.probabilities), dtype= float, requires_grad=True)
        self.eta = torch.tensor(0, dtype=float, requires_grad=True)
        self.u = torch.zeros(len(self.probabilities_ind), requires_grad=True)

        self.plot_d = [[] for i in range(len(self.probabilities))]
        self.plot_g = [[] for i in range(len(self.probabilities))]
        self.plot_u = [[] for i in range(len(self.probabilities))]
        self.plot_eta = []
        self.plot_q = [[[]for i in range(len(self.probabilities))] for i in range(self.community_size)]


        
class BasicFunctions:
    
    @staticmethod
    def utility_generation(agent: TorchPlayer) -> list:

        return [0.5 * agent.a * agent.G[proba] ** 2 
                + agent.b * agent.G[proba] 
                + agent.d for proba in agent.probabilities_ind]

    @staticmethod
    def utility_demand(agent: TorchPlayer) -> list:

        return [agent.a_tilde * (agent.D[proba] - agent.D_target[proba]) ** 2 
                - agent.b_tilde for proba in agent.probabilities_ind]

    @staticmethod
    def q_sum(agent: TorchPlayer, weights: bool = False) -> list:
        
        res = [torch.tensor(0, dtype=float) for proba in agent.probabilities_ind]
        
        #mask = torch.tensor(agent.connections, dtype=bool)

        #for proba in agent.probabilities_ind:
        #   
        #    actual_trades = agent.q.T[proba][mask]
        #    print(actual_trades)
        #
        #    if weights:
        #        res[proba] = torch.dot(actual_trades, torch.tensor(agent.trading_cost)[mask])
        #    else:
        #        res[proba] = actual_trades.sum()

        for proba in agent.probabilities_ind:
            for player, connection in enumerate(agent.connections):
                if connection:
                    if weights:
                        res[proba] += agent.trading_cost[player] * agent.q[player][proba]
                    else:
                        res[proba] += agent.q[player][proba]

        return res

    @staticmethod
    def utility_trading(agent: TorchPlayer) -> list:

        return BasicFunctions.q_sum(agent, weights= True)

    @staticmethod
    def utility(agent: TorchPlayer) -> list:
        
        demand = BasicFunctions.utility_demand(agent)
        generation = BasicFunctions.utility_generation(agent)
        trading = BasicFunctions.utility_trading(agent)

        return [demand[proba] + generation[proba] + trading[proba] for proba in agent.probabilities_ind]

    @staticmethod
    def penalty_demand_bounds_lower(agent: TorchPlayer) -> list:
        res = [torch.tensor(0, dtype=float) for proba in agent.probabilities_ind]

        for proba in agent.probabilities_ind:
            res[proba] = (agent.D_min - agent.D[proba]) ** 2 if (agent.D_min - agent.D[proba]) > 0 else torch.tensor(0, dtype=float)

        return res

    @staticmethod
    def penalty_demand_bounds_upper(agent: TorchPlayer) -> list:
        res = [torch.tensor(0, dtype=float) for proba in agent.probabilities_ind]

        for proba in agent.probabilities_ind:
            res[proba] = (agent.D[proba] - agent.D_max) ** 2 if (agent.D[proba] - agent.D_max) > 0 else torch.tensor(0, dtype=float)

        return res

    @staticmethod
    def penalty_generation_bounds_lower(agent: TorchPlayer) -> list:
        res = [torch.tensor(0, dtype=float) for proba in agent.probabilities_ind]

        for proba in agent.probabilities_ind:
            res[proba] = (agent.G_min - agent.G[proba]) ** 2 if (agent.G_min - agent.G[proba]) > 0 else torch.tensor(0, dtype=float)

        return res

    @staticmethod
    def penalty_generation_bounds_upper(agent: TorchPlayer) ->list:
        res = [torch.tensor(0, dtype=float) for proba in agent.probabilities_ind]

        for proba in agent.probabilities_ind:
            res[proba] = (agent.G[proba] - agent.G_max) ** 2 if (agent.G[proba] - agent.G_max) > 0 else torch.tensor(0, dtype=float)

        return res      

    @staticmethod
    def penalty_SD_balance(agent: TorchPlayer) -> list:
        res = [torch.tensor(0, dtype=float) for proba in agent.probabilities_ind]

        for proba in agent.probabilities_ind:
            res[proba] = (agent.D[proba] 
                        - agent.G[proba] 
                        - agent.G_res[proba] 
                        - BasicFunctions.q_sum(agent, weights=False)[proba])**2

        return res

    @staticmethod
    def penalty_trading_bound_upper(agent: TorchPlayer) -> list:
        res = [[torch.tensor(0, dtype=float) for proba in agent.probabilities_ind] for agent_2 in agent.connections]

        for agent_2, connection in enumerate(agent.connections):
            if connection:
                for proba in agent.probabilities_ind:
                    res[agent_2][proba] = ((agent.q[agent_2][proba] - agent.kappa[agent_2]) ** 2 
                                            if (agent.q[agent_2][proba] - agent.kappa[agent_2]) > 0 else torch.tensor(0, dtype=float))

        return res

    @staticmethod
    def penalty_trading_bound_lower(agent: TorchPlayer) -> list:
        res = [[torch.tensor(0, dtype=float) for proba in agent.probabilities_ind] for agent_2 in agent.connections]

        for agent_2, connection in enumerate(agent.connections):
            if connection:
                for proba in agent.probabilities_ind:
                    res[agent_2][proba] = ((- agent.kappa[agent_2] - agent.q[agent_2][proba]) ** 2 
                                        if (- agent.kappa[agent_2] - agent.q[agent_2][proba]) > 0 else torch.tensor(0, dtype=float))

        return res       

    @staticmethod
    def penalty_bilateral_trading_bounds(agent: TorchPlayer) -> list:
        res = [[torch.tensor(0, dtype=float) for proba in agent.probabilities_ind] for agent_2 in agent.connections]

        for agent_2, connection in enumerate(agent.connections):
            if connection:
                for proba in agent.probabilities_ind:
                    res[agent_2][proba] = (agent.q[agent_2][proba] + agent.q_others[agent_2][agent.id][proba]) ** 2

        return res 


class RiskProblemFunctions:

    @staticmethod
    def penalty_residual(agent: TorchPlayer) -> list:
        res = [torch.tensor(0, dtype=float) for proba in agent.probabilities_ind]

        for proba in agent.probabilities_ind:
            res[proba] = ((BasicFunctions.utility(agent)[proba] - agent.w[proba] - agent.j[proba] - agent.eta - agent.u[proba]) ** 2 
                            if (BasicFunctions.utility(agent)[proba] - agent.w[proba] - agent.j[proba] - agent.eta - agent.u[proba]) > 0 else torch.tensor(0, dtype = float))

        return res

    @staticmethod
    def insurance_bound_lower(agent: TorchPlayer) -> list:
        res = [torch.tensor(0, dtype=float) for proba in agent.probabilities_ind]

        for proba in agent.probabilities_ind:
            res[proba] = agent.j[proba] ** 2 if agent.j[proba] < 0 else torch.tensor(0, dtype=float)

        return res

    @staticmethod
    def insurance_bound_upper(agent: TorchPlayer) -> list:
        res = [torch.tensor(0, dtype=float) for proba in agent.probabilities_ind]

        for proba in agent.probabilities_ind:
            res[proba] = (agent.j[proba] - agent.j_max) ** 2 if (agent.j[proba] - agent.j_max) > 0 else torch.tensor(0, dtype=float)

        return res

    @staticmethod
    def residual_bound(agent: TorchPlayer) -> list:
        res = [torch.tensor(0, dtype=float) for proba in agent.probabilities_ind]

        for proba in agent.probabilities_ind:
            res[proba] = agent.u[proba] ** 2 if agent.u[proba] < 0 else torch.tensor(0, dtype=float)

        return res 

    @staticmethod
    def contract_trading_bound(agent: TorchPlayer) -> list:
        res = [torch.tensor(0, dtype=float) for proba in agent.probabilities_ind]

        for proba in agent.probabilities_ind:
            for agent_2 in range(agent.community_size):
                if agent_2 != agent.id:
                    res[proba] += agent.w_others[agent_2][proba]
                
            res[proba] += agent.w[proba]
            res[proba] = res[proba] ** 2

        return res

    @staticmethod
    def risk_utility_per_proba(agent: TorchPlayer):
        res = [torch.tensor(0, dtype=float) for proba in agent.probabilities_ind]

        for proba in agent.probabilities_ind:
            insurance_term = agent.j[proba] * agent.alpha[proba]
            contracts_term = agent.w[proba] * agent.gamma[proba]
            residual_term = agent.probabilities[proba] / (1 - agent.risk_aversion) * agent.u[proba] 

            res[proba] = insurance_term + contracts_term + residual_term

        return res