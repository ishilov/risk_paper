import gurobipy as gp
from pygments import lex

class Gurobi:

    @staticmethod
    def gurobi_add_generation_var(agent, model):

        for proba in agent.probabilities_ind:
            model.addVar(lb = agent.G_min,
                            ub = agent.G_max, 
                            vtype=gp.GRB.CONTINUOUS, 
                            name = f'G_{agent.id}_{proba}')

        model.update()

    @staticmethod
    def gurobi_add_demand_var(agent, model):

        for proba in agent.probabilities_ind:
            model.addVar(lb = agent.D_min,
                            ub = agent.D_max,
                            vtype = gp.GRB.CONTINUOUS,
                            name = f'D_{agent.id}_{proba}')

        model.update()

    @staticmethod
    def gurobi_add_energy_trading_var(agent, agents, model):

        for agent_2 in agents:
            if agent.connections[agent_2.id]:
                for proba in agent.probabilities_ind:
                        model.addVar(lb = - agent.kappa[agent_2.id],
                                    ub = agent.kappa[agent_2.id],
                                    vtype = gp.GRB.CONTINUOUS,
                                    name = f'q_{agent.id}_{agent_2.id}_{proba}')

        model.update()

    @staticmethod
    def gurobi_add_fin_contracts_var(agent, model):

        for proba in agent.probabilities_ind:
            model.addVar(lb = - float('inf'),
                        ub = float('inf'),
                        vtype = gp.GRB.CONTINUOUS,
                        name = f'W_{agent.id}_{proba}')

        model.update()

    @staticmethod
    def gurobi_add_insurance_var(agent, model):

        for proba in agent.probabilities_ind:
            model.addVar(lb = 0.0,
                        ub = agent.j_max,
                        vtype = gp.GRB.CONTINUOUS,
                        name = f'J_{agent.id}_{proba}')

        model.update()

    @staticmethod
    def gurobi_add_insurance_var_wht_bound(agent, model):

        for proba in agent.probabilities_ind:
            model.addVar(lb = 0.0,
                        ub = float('inf'),
                        vtype = gp.GRB.CONTINUOUS,
                        name = f'J_{agent.id}_{proba}')

        model.update()

    @staticmethod
    def gurobi_add_eta_var(agent, model):
        model.addVar(lb = - float('inf'),
                    ub = float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = f'eta_{agent.id}')

        model.update()

    @staticmethod
    def gurobi_add_residual_var(agent, model):
    
        for proba in agent.probabilities_ind:
            model.addVar(lb = 0,
                        ub = float('inf'),
                        vtype = gp.GRB.CONTINUOUS,
                        name = f'u_{agent.id}_{proba}')

        model.update()

    @staticmethod
    def gurobi_add_gamma_price_var(model, probabilities):
        for proba in probabilities:
            model.addVar(lb = - float('inf'),
                        ub = float('inf'),
                        vtype = gp.GRB.CONTINUOUS,
                        name = f'gamma_{proba}')

        model.update()

    @staticmethod
    def gurobi_set_objective(agent, model, price_as_var = False):
        if not price_as_var:
            lExpr = gp.LinExpr()
            for proba_ind, proba in enumerate(agent.probabilities):
                lExpr.add(model.getVarByName(f'W_{agent.id}_{proba_ind}') * agent.gamma[proba_ind] 
                        + model.getVarByName(f'J_{agent.id}_{proba_ind}') * agent.alpha[proba_ind]
                        + model.getVarByName(f'u_{agent.id}_{proba_ind}') * proba / (1 - agent.risk_aversion))

            lExpr.add(model.getVarByName(f'eta_{agent.id}'))

        else:
            lExpr = gp.QuadExpr()
            for proba_ind, proba in enumerate(agent.probabilities):
                lExpr.add(model.getVarByName(f'W_{agent.id}_{proba_ind}') * model.getVarByName(f'gamma_{proba_ind}')
                        + model.getVarByName(f'J_{agent.id}_{proba_ind}') * agent.alpha[proba_ind]
                        + model.getVarByName(f'u_{agent.id}_{proba_ind}') * proba / (1 - agent.risk_aversion))

            lExpr.add(model.getVarByName(f'eta_{agent.id}'))

        return lExpr

    @staticmethod
    def gurobi_set_objective_quadr_test(agent, model):
        qExpr = gp.QuadExpr()

        for proba_ind, proba in enumerate(agent.probabilities):
            qExpr.add(model.getVarByName(f'W_{agent.id}_{proba_ind}') * model.getVarByName(f'W_{agent.id}_{proba_ind}') * agent.gamma[proba_ind] 
                        + model.getVarByName(f'J_{agent.id}_{proba_ind}') * model.getVarByName(f'J_{agent.id}_{proba_ind}') * agent.alpha[proba_ind]
                        + model.getVarByName(f'u_{agent.id}_{proba_ind}') * model.getVarByName(f'u_{agent.id}_{proba_ind}') * proba / (1 - agent.risk_aversion))

        return qExpr

    
    @staticmethod
    def gurobi_trading_sum_calc(agent, proba, agents, model, weights = False):
        lExpr = gp.LinExpr()

        if weights:
            for agent_2 in agents:
                if agent.connections[agent_2.id]:
                    lExpr.add(model.getVarByName(f'q_{agent.id}_{agent_2.id}_{proba}') * agent.trading_cost[agent_2.id])

        else:
            for agent_2 in agents:
                if agent.connections[agent_2.id]:
                    lExpr.add(model.getVarByName(f'q_{agent.id}_{agent_2.id}_{proba}'))

        return lExpr

    @staticmethod
    def gurobi_set_SD_balance_constr(agent, agents, model):

        for proba in agent.probabilities_ind:
            model.addConstr(model.getVarByName(f'D_{agent.id}_{proba}')
                            - model.getVarByName(f'G_{agent.id}_{proba}')
                            - agent.G_res[proba]
                            - Gurobi.gurobi_trading_sum_calc(agent, proba, agents, model, weights=False) == 0,
                            name= f'SD balance for agent {agent.id} proba {proba}')

        model.update()

    @staticmethod
    def gurobi_set_bilateral_trading_constr(agent, agents, model):
        for agent_2 in agents:
            if agent.connections[agent_2.id]:
                for proba in agent.probabilities_ind:
                    model.addConstr(model.getVarByName(f'q_{agent.id}_{agent_2.id}_{proba}')  
                                    + model.getVarByName(f'q_{agent_2.id}_{agent.id}_{proba}') == 0, 
                                    name = f'Bilateral trading for pair ({agent.id}, {agent_2.id}) proba {proba}')

        model.update()

    @staticmethod
    def gurobi_quadr_generation(agent, proba, model):
        qExpr_G = gp.QuadExpr()

        qExpr_G.add(0.5 * agent.a 
                    * model.getVarByName(f'G_{agent.id}_{proba}') * model.getVarByName(f'G_{agent.id}_{proba}') 
                    + agent.b 
                    * model.getVarByName(f'G_{agent.id}_{proba}') 
                    + agent.d)
        
        return qExpr_G

    @staticmethod
    def gurobi_quadr_demand(agent, proba, model):
        qExpr_D = gp.QuadExpr()

        qExpr_D.add(agent.a_tilde 
                    * (model.getVarByName(f'D_{agent.id}_{proba}') - agent.D_target[proba]) 
                    * (model.getVarByName(f'D_{agent.id}_{proba}') - agent.D_target[proba]) 
                    - agent.b_tilde)

        return qExpr_D

    @staticmethod
    def gurobi_set_residual_constr(agent, agents, model):
    
        for proba in agent.probabilities_ind:
            qExpr = gp.QuadExpr()
            lExpr = gp.LinExpr()
            
            qExpr.add(Gurobi.gurobi_quadr_generation(agent, proba, model))
            qExpr.add(Gurobi.gurobi_quadr_demand(agent, proba, model))
            lExpr.add(Gurobi.gurobi_trading_sum_calc(agent, proba, agents, model, weights=True))

            lExpr.add(- model.getVarByName(f'u_{agent.id}_{proba}') )
            lExpr.add(- model.getVarByName(f'eta_{agent.id}'))
            lExpr.add(- model.getVarByName(f'W_{agent.id}_{proba}'))
            lExpr.add(- model.getVarByName(f'J_{agent.id}_{proba}'))

            model.addConstr(qExpr + lExpr  <= 0,
                            name=f'Residual constraint for agent {agent.id} proba {proba}')
            
            model.update()

    @staticmethod
    def gurobi_add_true_insurance_bound(agent, agents, model):
        for proba in agent.probabilities_ind:
            qExpr = gp.QuadExpr()
            lExpr = gp.LinExpr()
            
            qExpr.add(Gurobi.gurobi_quadr_generation(agent, proba, model))
            qExpr.add(Gurobi.gurobi_quadr_demand(agent, proba, model))
            lExpr.add(Gurobi.gurobi_trading_sum_calc(agent, proba, agents, model, weights=True))

            lExpr.add(- model.getVarByName(f'eta_{agent.id}'))

            model.addConstr(model.getVarByName(f'J_{agent.id}_{proba}') <= qExpr + lExpr,
                            name = f'Insurance constraint for agent {agent.id} proba {proba}')

        model.update()

    @staticmethod
    def gurobi_RN_utility(agent, agents, model):
        qExpr = gp.QuadExpr()

        for proba in agent.probabilities_ind:
            qExpr.add(agent.probabilities[proba] * Gurobi.gurobi_quadr_generation(agent, proba, model))
            qExpr.add(agent.probabilities[proba] * Gurobi.gurobi_quadr_demand(agent, proba, model))
            qExpr.add(agent.probabilities[proba] * Gurobi.gurobi_trading_sum_calc(agent, proba, agents, model, weights=True))

        return qExpr

    @staticmethod
    def gurobi_set_risk_trading_constr(agents, model):
        for proba in agents[0].probabilities_ind:
            lExpr = gp.LinExpr()

            for agent in agents:
                lExpr.add(model.getVarByName(f'W_{agent.id}_{proba}'))

            model.addConstr(lExpr == 0,
                            name = f'Risk trading balance for proba {proba}')

        model.update()

    @staticmethod
    def nullify_risk_trading(agent, model):
        for proba in agent.probabilities_ind:
            lExpr = gp.LinExpr()
            lExpr.add(model.getVarByName(f'W_{agent.id}_{proba}'))

            model.addConstr(lExpr == 0,
                            name = f'Risk trading zero for agent {agent.id} and proba {proba}')

        model.update()

    @staticmethod
    def nullify_insurance_trading(agent, model):
        for proba in agent.probabilities_ind:
            lExpr = gp.LinExpr()
            lExpr.add(model.getVarByName(f'J_{agent.id}_{proba}'))

            model.addConstr(lExpr == 0,
                            name = f'Insurance trading zero for agent {agent.id} and proba {proba}')

        model.update()
    

class BRGS:

    @staticmethod
    def pass_parameters(agent, agents):
        for agent_2 in agents:
            if agent.connections[agent_2.id]:
                agent.q_others.update({agent_2.id : agent_2.q[agent.id]})
            
            agent.w_others.update({agent_2.id : agent_2.w})

    @staticmethod
    def extract_parameters(agent, agents, model):
        for proba in agent.probabilities_ind:
            agent.G[proba] = model.getVarByName(f'G_{agent.id}_{proba}').X
            agent.D[proba] = model.getVarByName(f'D_{agent.id}_{proba}').X
            agent.j[proba] = model.getVarByName(f'J_{agent.id}_{proba}').X
            agent.w[proba] = model.getVarByName(f'W_{agent.id}_{proba}').X
            agent.u[proba] = model.getVarByName(f'u_{agent.id}_{proba}').X

            for agent_2 in agents:
                if agent.connections[agent_2.id]:
                    agent.q[agent_2.id][proba] = model.getVarByName(f'q_{agent.id}_{agent_2.id}_{proba}').X

        agent.eta = model.getVarByName(f'eta_{agent.id}').X

    @staticmethod
    def brgs_gurobi_set_bilateral_trading_constr(agent, agents, model):
        for agent_2 in agents:
            if agent.connections[agent_2.id]:
                for proba in agent.probabilities_ind:
                    model.addConstr(model.getVarByName(f'q_{agent.id}_{agent_2.id}_{proba}')  
                                    + agent.q_others[agent_2.id][proba] <= 0, 
                                    name = f'Bilateral trading for pair ({agent.id}, {agent_2.id}) proba {proba}')

        model.update()

    @staticmethod
    def brgs_gurobi_set_risk_trading_constr(agent, agents, model):        
        for proba in agent.probabilities_ind:
            lExpr = gp.LinExpr()

            lExpr.add(model.getVarByName(f'W_{agent.id}_{proba}'))
            sum_others = sum([agent.w_others[agent_2.id][proba] for agent_2 in agents])

            model.addConstr(lExpr + sum_others == 0,
                            name = f'Risk trading balance for agent {agent.id} for proba {proba}')

        model.update()


class GurobiSolution(Gurobi, BRGS):
    def __init__(self, agents, model, solution_type='centralized', agent = None) -> None:

        self.agents = agents
        self.model = model
        
        if solution_type in ('centralized_pessimistic', 'centralized_optimistic', 
                            'centralized_without_finance', 'risk-neutral',
                            'BRGS', 'initial', 'test', 'quadratic_test',
                            'centralized_true_insurance_constraint',
                            'without_IC'):

            self.solution_type = solution_type

            if solution_type == 'BRGS':
                self.agent = agent

    def build_model(self, price_as_var = False):
        if self.solution_type == 'test':
            for agent in self.agents:
                Gurobi.gurobi_add_generation_var(agent, self.model)
                Gurobi.gurobi_add_demand_var(agent, self.model)
                Gurobi.gurobi_add_energy_trading_var(agent, self.agents, self.model)

            for agent in self.agents:
                Gurobi.gurobi_set_SD_balance_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_bilateral_trading_constr(agent, self.agents, self.model)

            obj = gp.QuadExpr()
            for agent in self.agents:
                for proba in agent.probabilities_ind:
                    obj.add(Gurobi.gurobi_quadr_demand(agent, proba, self.model))
                    obj.add(Gurobi.gurobi_quadr_generation(agent, proba, self.model))
                    obj.add(Gurobi.gurobi_trading_sum_calc(agent, proba, self.agents, self.model, weights=True))

            self.model.setObjective(obj, gp.GRB.MINIMIZE)


        if self.solution_type == 'initial':
            for agent in self.agents:
                Gurobi.gurobi_add_demand_var(agent, self.model)
                Gurobi.gurobi_add_generation_var(agent, self.model)
                Gurobi.gurobi_add_energy_trading_var(agent, self.agents, self.model)

            for agent in self.agents:
                Gurobi.gurobi_set_bilateral_trading_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_SD_balance_constr(agent, self.agents, self.model)

            obj = gp.QuadExpr()
            for agent in self.agents:
                for proba in agent.probabilities_ind:
                
                    obj.add(Gurobi.gurobi_quadr_generation(agent, proba, self.model))
                    obj.add(Gurobi.gurobi_quadr_demand(agent, proba, self.model))
                    obj.add(Gurobi.gurobi_trading_sum_calc(agent, proba, self.agents, self.model, weights=True))

            self.model.setObjective(obj, gp.GRB.MINIMIZE)

        if self.solution_type == 'centralized_without_finance':
            for agent in self.agents:
                Gurobi.gurobi_add_demand_var(agent, self.model)
                Gurobi.gurobi_add_generation_var(agent, self.model)
                Gurobi.gurobi_add_energy_trading_var(agent, self.agents, self.model)
                Gurobi.gurobi_add_eta_var(agent, self.model)
                Gurobi.gurobi_add_residual_var(agent, self.model)
                Gurobi.gurobi_add_fin_contracts_var(agent, self.model)
                Gurobi.gurobi_add_insurance_var(agent, self.model)

            for agent in self.agents:
                Gurobi.nullify_insurance_trading(agent, self.model)
                Gurobi.gurobi_set_bilateral_trading_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_residual_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_SD_balance_constr(agent, self.agents, self.model)
                Gurobi.nullify_risk_trading(agent, self.model)

            Gurobi.gurobi_set_risk_trading_constr(self.agents, self.model)

            if not price_as_var:
                obj = gp.LinExpr()
                for agent in self.agents:
                    obj.add(Gurobi.gurobi_set_objective(agent, self.model, price_as_var))   


        if self.solution_type == 'without_IC':
            for agent in self.agents:
                Gurobi.gurobi_add_demand_var(agent, self.model)
                Gurobi.gurobi_add_generation_var(agent, self.model)
                Gurobi.gurobi_add_energy_trading_var(agent, self.agents, self.model)
                Gurobi.gurobi_add_eta_var(agent, self.model)
                Gurobi.gurobi_add_fin_contracts_var(agent, self.model)
                Gurobi.gurobi_add_insurance_var(agent, self.model)
                Gurobi.gurobi_add_residual_var(agent, self.model)    
                Gurobi.nullify_insurance_trading(agent, self.model)
                

            for agent in self.agents:
                Gurobi.gurobi_set_bilateral_trading_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_residual_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_SD_balance_constr(agent, self.agents, self.model)
                
            Gurobi.gurobi_set_risk_trading_constr(self.agents, self.model)
        
            if not price_as_var:
                obj = gp.LinExpr()
                for agent in self.agents:
                    obj.add(Gurobi.gurobi_set_objective(agent, self.model, price_as_var))   

        if self.solution_type == 'centralized_true_insurance_constraint':
            for agent in self.agents:
                Gurobi.gurobi_add_demand_var(agent, self.model)
                Gurobi.gurobi_add_generation_var(agent, self.model)
                Gurobi.gurobi_add_energy_trading_var(agent, self.agents, self.model)
                Gurobi.gurobi_add_eta_var(agent, self.model)
                Gurobi.gurobi_add_fin_contracts_var(agent, self.model)
                Gurobi.gurobi_add_insurance_var_wht_bound(agent, self.model)
                Gurobi.gurobi_add_residual_var(agent, self.model)
                

            for agent in self.agents:
                Gurobi.gurobi_set_bilateral_trading_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_residual_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_SD_balance_constr(agent, self.agents, self.model)
                Gurobi.gurobi_add_true_insurance_bound(agent, self.agents, self.model)
                
            Gurobi.gurobi_set_risk_trading_constr(self.agents, self.model)

            epsilon = 1e-4
        
            if not price_as_var:
                obj = gp.LinExpr()
                for agent in self.agents:
                    obj.add(Gurobi.gurobi_set_objective(agent, self.model, price_as_var))
            

        if self.solution_type == 'centralized_pessimistic':
            for agent in self.agents:
                Gurobi.gurobi_add_demand_var(agent, self.model)
                Gurobi.gurobi_add_generation_var(agent, self.model)
                Gurobi.gurobi_add_energy_trading_var(agent, self.agents, self.model)
                Gurobi.gurobi_add_eta_var(agent, self.model)
                Gurobi.gurobi_add_fin_contracts_var(agent, self.model)
                Gurobi.gurobi_add_insurance_var(agent, self.model)
                Gurobi.gurobi_add_residual_var(agent, self.model)
                

            for agent in self.agents:
                Gurobi.gurobi_set_bilateral_trading_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_residual_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_SD_balance_constr(agent, self.agents, self.model)
                
            Gurobi.gurobi_set_risk_trading_constr(self.agents, self.model)

            epsilon = 1e-4
        
            if not price_as_var:
                obj = gp.LinExpr()
                for agent in self.agents:
                    obj.add(Gurobi.gurobi_set_objective(agent, self.model, price_as_var))
            
            else:
                obj = gp.QuadExpr()
                for agent in self.agents:
                    Gurobi.gurobi_add_gamma_price_var(self.model, self.agents[0].probabilities_ind)
                    obj.add(Gurobi.gurobi_set_objective(agent, self.model, price_as_var))

            self.model.setObjective(obj, gp.GRB.MINIMIZE)

        if self.solution_type == 'risk-neutral':
            obj = gp.QuadExpr()

            for agent in self.agents:
                Gurobi.gurobi_add_demand_var(agent, self.model)
                Gurobi.gurobi_add_generation_var(agent, self.model)
                Gurobi.gurobi_add_energy_trading_var(agent, self.agents, self.model)

            for agent in self.agents:
                Gurobi.gurobi_set_bilateral_trading_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_SD_balance_constr(agent, self.agents, self.model)
                
            for agent in self.agents:    
                obj_agents = Gurobi.gurobi_RN_utility(agent, self.agents, self.model)
                obj.add(obj_agents)
               
            self.model.setObjective(obj, gp.GRB.MINIMIZE)

        if self.solution_type == 'centralized_optimistic':
            for agent in self.agents:
                Gurobi.gurobi_add_demand_var(agent, self.model)
                Gurobi.gurobi_add_generation_var(agent, self.model)
                Gurobi.gurobi_add_energy_trading_var(agent, self.agents, self.model)
                Gurobi.gurobi_add_eta_var(agent, self.model)
                Gurobi.gurobi_add_fin_contracts_var(agent, self.model)
                Gurobi.gurobi_add_insurance_var(agent, self.model)
                Gurobi.gurobi_add_residual_var(agent, self.model)
                Gurobi.nullify_risk_trading(agent, self.model)
                

            for agent in self.agents:
                Gurobi.gurobi_set_bilateral_trading_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_residual_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_SD_balance_constr(agent, self.agents, self.model)
                
            Gurobi.gurobi_set_risk_trading_constr(self.agents, self.model)

            epsilon = 1e-4
        
            if not price_as_var:
                obj = gp.LinExpr()
                for agent in self.agents:
                    obj.add(Gurobi.gurobi_set_objective(agent, self.model, price_as_var))
            
            else:
                obj = gp.QuadExpr()
                for agent in self.agents:
                    Gurobi.gurobi_add_gamma_price_var(self.model, self.agents[0].probabilities_ind)
                    obj.add(Gurobi.gurobi_set_objective(agent, self.model, price_as_var))

            self.model.setObjective(obj, gp.GRB.MINIMIZE)

        if self.solution_type == 'quadratic_test':
            for agent in self.agents:
                Gurobi.gurobi_add_demand_var(agent, self.model)
                Gurobi.gurobi_add_generation_var(agent, self.model)
                Gurobi.gurobi_add_energy_trading_var(agent, self.agents, self.model)
                Gurobi.gurobi_add_eta_var(agent, self.model)
                Gurobi.gurobi_add_fin_contracts_var(agent, self.model)
                Gurobi.gurobi_add_insurance_var(agent, self.model)
                Gurobi.gurobi_add_residual_var(agent, self.model)
                

            for agent in self.agents:
                Gurobi.gurobi_set_bilateral_trading_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_residual_constr(agent, self.agents, self.model)
                Gurobi.gurobi_set_SD_balance_constr(agent, self.agents, self.model)
                
            Gurobi.gurobi_set_risk_trading_constr(self.agents, self.model)

            epsilon = 1e-4
        
            if not price_as_var:
                obj = gp.QuadExpr()
                for agent in self.agents:
                    obj.add(Gurobi.gurobi_set_objective_quadr_test(agent, self.model))

            self.model.setObjective(obj, gp.GRB.MINIMIZE)
            
        if self.solution_type == 'BRGS': 
            Gurobi.gurobi_add_demand_var(self.agent, self.model)
            Gurobi.gurobi_add_generation_var(self.agent, self.model)
            Gurobi.gurobi_add_energy_trading_var(self.agent, self.agents, self.model)
            Gurobi.gurobi_add_eta_var(self.agent, self.model)
            Gurobi.gurobi_add_fin_contracts_var(self.agent, self.model)
            Gurobi.gurobi_add_insurance_var(self.agent, self.model)
            Gurobi.gurobi_add_residual_var(self.agent, self.model)

            BRGS.brgs_gurobi_set_bilateral_trading_constr(self.agent, self.agents, self.model)
            BRGS.brgs_gurobi_set_risk_trading_constr(self.agent, self.agents, self.model)
            Gurobi.gurobi_set_residual_constr(self.agent, self.agents, self.model)
            Gurobi.gurobi_set_SD_balance_constr(self.agent, self.agents, self.model)

            obj = gp.LinExpr()
            obj.add(Gurobi.gurobi_set_objective(self.agent, self.model, two_level=True))
            self.model.setObjective(obj, gp.GRB.MINIMIZE)

        self.model.update()