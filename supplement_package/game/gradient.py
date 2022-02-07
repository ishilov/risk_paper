from supplement_package.game.player import Player
import numpy as np
from typing import List

class GradientComputation:

    @staticmethod
    def utility_grad(player: Player) -> dict:

        update_eta = 1

        update_u = np.array([proba/(1-player.risk_aversion) for proba in player.probabilities])

        return {'update_eta': update_eta,
                'update_u': update_u}
    

    @staticmethod
    def penalty_dmin(player: Player) -> np.ndarray:
        """Returns array with the values (D_min - D^w) for scenarios w \in W in which
        D_min - D^w >=0 and with 0 otherwise"""

        return np.array([-player.D_min + player.D[proba] if player.D_min - player.D[proba] >=0 else 0 for proba in player.probabilities_ind])

        

    
    @staticmethod
    def penalty_dmax(player: Player) -> np.ndarray:
        """Returns array with the values (D^w - D_max) for scenarios w \in W in which
        D^w - D_max >=0 and with 0 otherwise"""

        return np.array([player.D[proba] - player.D_max if player.D[proba] - player.D_max >=0 else 0 for proba in player.probabilities_ind])


    @staticmethod
    def penalty_gmin(player: Player) -> np.ndarray:
        """Returns array with the values (G_min - G^w) for scenarios w \in W in which
        G_min - G^w >=0 and with 0 otherwise"""

        return np.array([-player.G_min + player.G[proba] if player.G_min - player.G[proba] >=0 else 0 for proba in player.probabilities_ind])


    @staticmethod
    def penalty_gmax(player: Player) -> np.ndarray:
        """Returns array with the values (G^w - G_max) for scenarios w \in W in which
        G^w - G_max>=0 and with 0 otherwise"""

        return np.array([player.G[proba] - player.G_max if player.G[proba] - player.G_max >=0 else 0 for proba in player.probabilities_ind])


    @staticmethod
    def penalty_trading_bound(player: Player) -> np.ndarray:
        """For each neighbor m in the player's connections computes the difference q^w_{nm} - kappa_{nm}
        and returns 2d-array with q^w_{nm} - kappa_{nm} if q^w_{nm} - kappa_{nm} >=0 and 0 otherwise"""

        res = np.zeros_like(player.q,  dtype= float)
        
        for neighbor in range(len(player.connections)):
            for proba in player.probabilities_ind:
                res[neighbor][proba] = player.q[neighbor][proba] - player.kappa[neighbor] if player.q[neighbor][proba] - player.kappa[neighbor] >=0 else 0

        return res


    @staticmethod
    def penalty_balance(player: Player) -> dict:
        """This function computes an update for the gradient from the penalty function associated with the supply-demand balance equation.
        
        D^w_n - \Delta G^w_n - G^w_n - \sum_{m \in \Gamma_n} q^w_{nm} = 0

        Updated values are

        - grad_D
        - grad_G 
        - grad_q

        Returns a dictionary with the names of the updated variables as keys and corresponding np.ndarrays as values"""

        update_d = player.D - player.G_res - player.G - player.q_sum()

        update_g = -1*(player.D - player.G_res - player.G - player.q_sum())

        update_q = np.zeros_like(player.q, dtype= float)


        for neighbor in range(len(player.connections)):
            update_q[neighbor] = -1*(player.D - player.G_res - player.G - player.q_sum()) if player.connections[neighbor] else np.zeros_like(player.probabilities_ind)

        return {'update_d' : update_d,
                'update_g' : update_g,
                'update_q' : update_q,
                'violation' : player.D - player.G_res - player.G - player.q_sum()
                }


    @staticmethod
    def penalty_u(player: Player) -> np.ndarray:
        """This function checks the condition 
        
        0 <= u^w_n

        and returns an array with the values -u^w_n if -u^w_n >=0 (when condition is violated) and 0 otherwise"""

        return np.array([player.u[proba] if -player.u[proba] >=0 else 0 for proba in player.probabilities_ind])


    @staticmethod
    def penalty_residual(player: Player) -> dict:
        """This function returns a dict with the keys for the names of the updates and the values are updates themselves.
        The penalty function considered corresponds to the constraint
        
        \Pi^w_n - eta_n - u^w_n <=0

        Where \Pi^w_n is the utility (initial one) of agent n in the scenario w"""

        constraint = np.array([player.utility()[proba] - player.eta - player.u[proba] 
                            if player.utility()[proba] - player.eta - player.u[proba] >=0 else 0 for proba in player.probabilities_ind])

        update_d = 2 * player.a_tilde * (player.D - player.D_target) * constraint

        update_g = (2 * player.a * player.G + player.b) * constraint

        update_eta = sum(-1 * constraint)

        update_u = -1 * constraint

        update_q = np.zeros_like(player.q, dtype= float)

        for neighbor in range(len(player.connections)):
            update_q[neighbor] = player.trading_cost[neighbor]*constraint if player.connections[neighbor] else np.zeros_like(player.probabilities_ind)

        return {'update_d': update_d,
                'update_g': update_g,
                'update_eta': update_eta,
                'update_u': update_u,
                'update_q': update_q}



    @staticmethod
    def penalty_bilateral_trading(player: Player, neighbors: List[Player]) -> np.ndarray:
        """Using the function one_neighbor this function returns an 2d-array 
        where each line represents the trades with one neighbor m with the values
        
        q^w_{nm} + q^w_{mn} if q^w_{nm} + q^w_{mn} >=0 for w \in W and 0 otherwise
        
        Also it fills the line m with zeros if m is not a neighbor of n"""
        
        def one_neighbor(player: Player, neighbor: Player) -> np.ndarray:
            """For a given neighbor m of agent n this function returns 
            
            q^w_{nm} + q^w_{mn} if q^w_{nm} + q^w_{mn} >=0 for w \in W and 0 otherwise

            Also it returns array of zeros if agent m is not in the connections of the agent n"""

            if player.connections[neighbor.id]:
                return np.array([player.q[neighbor.id][proba] + neighbor.q[player.id][proba] 
                                if player.q[neighbor.id][proba] + neighbor.q[player.id][proba] >= 0 else 0 for proba in player.probabilities_ind])
            else:
                return np.zeros_like(player.probabilities_ind)

        res = np.zeros_like(player.q, dtype= float)

        for neighbor in neighbors:
            res[neighbor.id] = one_neighbor(player, neighbor)

        return res