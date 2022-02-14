import gurobipy as gp

class Gurobi:

    def __init__(self,
                agents,
                model,
                solution_type = 'centralized') -> None:
        
        self.agents = agents
        self.model = model
        
        if solution_type in ('centralized'):
            self.solution_type = solution_type
    
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
    def gurobi_set_objective(agent, model, two_level = True):
        if two_level:
            lExpr = gp.LinExpr()
            for proba_ind, proba in enumerate(agent.probabilities):
                lExpr.add(model.getVarByName(f'W_{agent.id}_{proba_ind}') * agent.gamma[proba_ind] 
                        + model.getVarByName(f'J_{agent.id}_{proba_ind}') * agent.alpha[proba_ind]
                        + model.getVarByName(f'u_{agent.id}_{proba_ind}') * proba / (1 - agent.risk_aversion))

            lExpr.add(model.getVarByName(f'eta_{agent.id}'))

        return lExpr

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
    def gurobi_set_risk_trading_constr(agents, model):
        for proba in agents[0].probabilities_ind:
            lExpr = gp.LinExpr()

            for agent in agents:
                lExpr.add(model.getVarByName(f'W_{agent.id}_{proba}'))

            model.addConstr(lExpr == 0,
                            name = f'Risk trading balance for proba {proba}')

        model.update()

    def build_model(self):
        if self.solution_type == 'centralized':
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

            obj = gp.LinExpr()
            for agent in self.agents:
                obj.add(Gurobi.gurobi_set_objective(agent, self.model, two_level=True))

            self.model.setObjective(obj, gp.GRB.MINIMIZE)
            self.model.update()
    

class BRGS(Gurobi):
    def __init__(self, agents, model, solution_type='centralized') -> None:
        super().__init__(agents, model, solution_type)

    def pass_parameter():
        pass

    