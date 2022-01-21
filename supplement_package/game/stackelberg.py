from game.player import Player
from game.gradient import GradientComputation

import numpy as np

class StackelbergPlayer(Player):
    
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
                alpha: list,
                insurance_bound: list) -> None:

        super().__init__(id, 
                        D_target, 
                        G_res, 
                        a, b, d, 
                        a_tilde, b_tilde, 
                        D_min, D_max, 
                        G_min, G_max, 
                        risk_aversion, kappa, trading_cost, connections)

        self.alpha = alpha
        self.j_max = insurance_bound

        self.j = np.zeros_like(self.probabilities, dtype= float)

    def risk_utility(self) -> np.ndarray:
        return self.utility() - self.j


class StackelbergGradientComputation(GradientComputation):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def utility_contr_grad(player: StackelbergPlayer) -> dict:
        update_eta = 1

        update_u = np.array([proba/(1-player.risk_aversion) for proba in player.probabilities])

        update_j = np.array([-player.alpha[proba] for proba in player.probabilities_ind])

        return {'update_eta': update_eta,
                'update_u': update_u,
                'update_j': update_j}

    @staticmethod
    def stackelberg_penalty_residual(player: StackelbergPlayer) -> dict:
        """This function returns a dict with the keys for the names of the updates and the values are updates themselves.
        The penalty function considered corresponds to the constraint
        
        \Pi^w_n - J^w_n - eta_n - u^w_n <=0

        Where \Pi^w_n is the utility (initial one) of agent n in the scenario w"""

        constraint = np.array([player.risk_utility()[proba] - player.j - player.eta - player.u[proba] 
                            if player.risk_utility()[proba] - player.j - player.eta - player.u[proba] >=0 else 0 for proba in player.probabilities_ind])

        update_d = 2 * player.a_tilde * (player.D - player.D_target) * constraint

        update_g = (2 * player.a * player.G + player.b) * constraint

        update_eta = sum(-1 * constraint)

        update_u = -1 * constraint

        update_q = np.zeros_like(player.q, dtype= float)

        update_j = 1 * constraint

        for neighbor in range(len(player.connections)):
            update_q[neighbor] = player.trading_cost[neighbor]*constraint if player.connections[neighbor] else np.zeros_like(player.probabilities_ind)

        return {'update_d': update_d,
                'update_g': update_g,
                'update_eta': update_eta,
                'update_u': update_u,
                'update_q': update_q,
                'update_j' :update_j}

    @staticmethod
    def penalty_jmin(player: StackelbergPlayer) -> np.ndarray:
        return np.array([player.j[proba] if -player.j[proba] >= 0 else 0 for proba in player.probabilities_ind])

    @staticmethod
    def penalty_jmax(player: StackelbergPlayer) -> np.ndarray:
        return np.array([player.j[proba] - player.j_max if player.j[proba] - player.j_max >=0 else 0 for proba in player.probabilities_ind])


class InsuranceCompany():

    def __init__(self,
                risk_attitude: float,
                probabilities: list,
                players: list) -> None:
        
        self.risk_attitude = risk_attitude
        
        self.eta = 0
        self.probabilities = probabilities
        self.u = np.zeros((len(players), len(probabilities)))
        self.alpha = np.zeros((len(players), len(probabilities)))

        self.j_max = np.zeros(len(probabilities))



class ICGradientComputation():

    @staticmethod
    def utility_grad(company: InsuranceCompany, players: list):
        update_alpha = np.zeros_like(company.alpha)
        update_u = np.zeros_like(company.u)
        
        for player in players:
            for i, proba in enumerate(company.probabilities):
                update_u[player.id][i] = proba / (1 - company.risk_attitude) 
                update_alpha[player.id][i] = - player.j[i]

        update_eta = len(players) 

        return {'update_alpha': update_alpha,
                'update_eta': update_eta,
                'update_u': update_u}

    @staticmethod
    def penalty_jmin(company: InsuranceCompany, players: list):
        pass

    @staticmethod
    def penalty_jmax(company: InsuranceCompany, players: list):
        return np.array([[player.j[proba] - company.j_max[player.id][proba] 
                        if player.j[proba] - company.j_max[player.id][proba] >=0 else 0
                        for proba in range(len(company.probabilities))] for player in players])

    @staticmethod
    def penlaty_u(company: InsuranceCompany, players: list):
        return np.array([[company.u[player.id][proba] if -company.u[player.id][proba] >=0 else 0
                        for proba in range(len(company.probabilities))] for player in players])

    @staticmethod
    def penalty_residual(company: InsuranceCompany, players: list):
        
        constraint = np.zeros_like(company.u)

        for player in players:
            for i, proba in enumerate(company.probabilities):
                constraint[player.id][i] = (player.j[i] - company.eta - company.u[player.id][i]
                                            if player.j[i] - company.eta - company.u[player.id][i] >=0 else 0)

        update_eta = -1 * constraint.sum()

        update_u = -1 * constraint

        return {'update_eta': update_eta,
                'update_u': update_u}