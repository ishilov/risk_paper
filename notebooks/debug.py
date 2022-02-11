import pandas as pd
import numpy as np
import random
import scipy.optimize as scopt

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
from supplement_package.game.gradient import GradientComputation
from supplement_package.game.player import Player
from supplement_package.game.stackelberg import StackelbergPlayer
from supplement_package.game.stackelberg import StackelbergGradientComputation
StackelbergGradientComputation.__dict__
community_size = 3
if community_size == 3:
    D_min = [0.0,0.0,0.0]
    D_max = [10.0,10.0,10.0]

    G_min = [0.0,0.0,0.0]
    G_max = [10.0,0.0,0.0]
    
    Kappa = [[0.0, 10.0, 10.0],
             [10.0, 0.0, 5.0],
             [10.0, 5.0, 0.0]]
    
    Cost = [[0.0, 1.0, 1.0],
            [3.0, 0.0, 1.0],
            [2.0, 1.0, 0.0]]
    
    probabilities = [0.333, 0.333, 0.333]

    #probabilities = [0.5, 0.5]
    connection_matrix = [[0,1,1],[1,0,1],[1,1,0]]
A_tilde = [random.uniform(0,1) for i in range(community_size)]
B_tilde = [random.uniform(0,1) for i in range(community_size)]

a = [random.uniform(0,1) for i in range(community_size)]
b = [random.uniform(0,1) for i in range(community_size)]
d = [random.uniform(0,1) for i in range(community_size)]

d_target = [[random.uniform(0,8) for j in range(len(probabilities))] for i in range(community_size)]
g_res = [[random.uniform(0,3) for j in range(len(probabilities))] for i in range(community_size)]

g_res = np.array(g_res)
d_target = np.array(d_target)

risk_aversion = [random.uniform(0,1) for i in range(community_size)]
agents = []
StackelbergPlayer.community_size = community_size
StackelbergPlayer.probabilities = probabilities

epsilon = 0.001
alpha = [[proba/(1 - risk_aversion[i]) - epsilon for proba in probabilities] for i in range(community_size)]
#alpha = [[0.2 for proba in probabilities] for i in range(community_size)]
gamma = [proba/(1 - min(risk_aversion)) for proba in probabilities]

j_max = [10 for i in range(community_size)]

for i in range(community_size):
    agent = StackelbergPlayer(i, d_target[i], g_res[i], a[i], b[i], d[i], 
                A_tilde[i], B_tilde[i], D_min[i], D_max[i], 
                G_min[i], G_max[i], risk_aversion[i], Kappa[i], Cost[i], connection_matrix[i],
                alpha = alpha[i], 
                gamma = gamma, 
                insurance_bound=10)
    
    agents.append(agent)
## Gurobi
import gurobipy as gp
def gurobi_add_generation_var(agent, model):

    for proba in agent.probabilities_ind:
        model.addVar(lb = agent.G_min, 
                        ub = agent.G_max, 
                        vtype=gp.GRB.CONTINUOUS, 
                        name = f'G_{agent.id}_{proba}')

    model.update()
def gurobi_add_demand_var(agent, model):

    for proba in agent.probabilities_ind:
        model.addVar(lb = agent.D_min,
                        ub = agent.D_max,
                        vtype = gp.GRB.CONTINUOUS,
                        name = f'D_{agent.id}_{proba}')

    model.update()
def gurobi_add_energy_trading_var(agent, agents, model):

    for agent_2 in agents:
        if agent.connections[agent_2.id]:
            for proba in agent.probabilities_ind:
                    model.addVar(lb = - agent.kappa[agent_2.id],
                                ub = agent.kappa[agent_2.id],
                                vtype = gp.GRB.CONTINUOUS,
                                name = f'q_{agent.id}_{agent_2.id}_{proba}')

    model.update()

def gurobi_add_fin_contracts_var(agent, model):

    for proba in agent.probabilities_ind:
        model.addVar(lb = - float('inf'),
                    ub = float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = f'W_{agent.id}_{proba}')

    model.update()
def gurobi_add_insurance_var(agent, model):

    for proba in agent.probabilities_ind:
        model.addVar(lb = 0.0,
                    ub = agent.j_max,
                    vtype = gp.GRB.CONTINUOUS,
                    name = f'J_{agent.id}_{proba}')

    model.update()
def gurobi_add_eta_var(agent, model):
    model.addVar(lb = - float('inf'),
                ub = float('inf'),
                vtype = gp.GRB.CONTINUOUS,
                name = f'eta_{agent.id}')

    model.update()
def gurobi_add_residual_var(agent, model):
    
    for proba in agent.probabilities_ind:
        model.addVar(lb = 0,
                    ub = float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = f'u_{agent.id}_{proba}')

    model.update()
def gurobi_set_objective(agent, model, two_level = True):
    if two_level:
        lExpr = gp.LinExpr()

        for proba_ind, proba in enumerate(agent.probabilities):
            lExpr.add(model.getVarByName(f'W_{agent.id}_{proba_ind}') * gamma[proba_ind] 
                    + model.getVarByName(f'J_{agent.id}_{proba_ind}') * alpha[agent.id][proba_ind]
                    + model.getVarByName(f'u_{agent.id}_{proba_ind}') * proba / (1 - agent.risk_aversion))

        lExpr.add(model.getVarByName(f'eta_{agent.id}'))

    return lExpr
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
def gurobi_set_SD_balance_constr(agent, agents, model):

    for proba in agent.probabilities_ind:
        model.addConstr(model.getVarByName(f'D_{agent.id}_{proba}')
                        - model.getVarByName(f'G_{agent.id}_{proba}')
                        - agent.G_res[proba]
                        - gurobi_trading_sum_calc(agent, proba, agents, model, weights=False) == 0,
                        name= f'SD balance for agent {agent.id} proba {proba}')

    model.update()
def gurobi_set_bilateral_trading_constr(agent, agents, model):
    for agent_2 in agents:
        if agent.connections[agent_2.id]:
            for proba in agent.probabilities_ind:
                model.addConstr(model.getVarByName(f'q_{agent.id}_{agent_2.id}_{proba}')  
                                + model.getVarByName(f'q_{agent_2.id}_{agent.id}_{proba}') == 0, 
                                name = f'Bilateral trading for pair ({agent.id}, {agent_2.id}) proba {proba}')

    model.update()
def gurobi_quadr_generation(agent, proba, model):
    qExpr_G = gp.QuadExpr()

    qExpr_G.add(0.5 * agent.a 
                * model.getVarByName(f'G_{agent.id}_{proba}') * model.getVarByName(f'G_{agent.id}_{proba}') 
                + agent.b 
                * model.getVarByName(f'G_{agent.id}_{proba}') 
                + agent.d)
    
    return qExpr_G
def gurobi_quadr_demand(agent, proba, model):
    qExpr_D = gp.QuadExpr()

    qExpr_D.add(agent.a_tilde 
                * (model.getVarByName(f'D_{agent.id}_{proba}') - agent.D_target[proba]) 
                * (model.getVarByName(f'D_{agent.id}_{proba}') - agent.D_target[proba]) 
                - agent.b_tilde)

    return qExpr_D
def gurobi_set_residual_constr(agent, agents, model):
    
    for proba in agent.probabilities_ind:
        model.addConstr(gurobi_quadr_generation(agent, proba, model) 
                        + gurobi_quadr_demand(agent, proba, model)
                        + gurobi_trading_sum_calc(agent, proba, agents, model, weights=True)
                        - model.getVarByName(f'u_{agent.id}_{proba}')
                        - model.getVarByName(f'eta_{agent.id}') <= 0,
                        name=f'Residual constraint for agent {agent.id} proba {proba}')

    model.update()

def gurobi_set_risk_trading_constr(agents, model):
    for proba in agents[0].probabilities_ind:
        lExpr = gp.LinExpr()

        for agent in agents:
            lExpr.add(model.getVarByName(f'W_{agent.id}_{proba}'))

        model.addConstr(lExpr == 0,
                        name = f'Risk trading balance for proba {proba}')

    model.update()
def build_model(agents, model, centralized = True):
    if centralized:
        for agent in agents:
            gurobi_add_demand_var(agent, model)
            gurobi_add_generation_var(agent, model)
            gurobi_add_energy_trading_var(agent, agents, model)
            gurobi_add_eta_var(agent, model)
            gurobi_add_fin_contracts_var(agent, model)
            gurobi_add_insurance_var(agent, model)
            gurobi_add_residual_var(agent, model)

        for agent in agents:
            gurobi_set_bilateral_trading_constr(agent, agents, model)
            gurobi_set_residual_constr(agent, agents, model)
            gurobi_set_SD_balance_constr(agent, agents, model)
            
        gurobi_set_risk_trading_constr(agents, model)

        obj = gp.LinExpr()
        for agent in agents:
            obj.add(gurobi_set_objective(agent, model, two_level=True))

        model.setObjective(obj, gp.GRB.MINIMIZE)
        model.update()
model_1 = gp.Model()
build_model(agents, model_1)

model_1.getConstrs()