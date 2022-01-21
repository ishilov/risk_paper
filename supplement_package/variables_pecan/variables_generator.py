import numpy as np
import variables_storage
import random

class VariablesGenerator:

#    def __init__(self,
#                scenario_amount: int,
##                agents_amount = 25,
#                generate_new = False,
#                ) -> None:
        
#        self.scenario_amount = scenario_amount
#        self.agents_amount = agents_amount
#        self.generate_new = generate_new
    @staticmethod
    def generator(scenario_amount,
                storage: variables_storage.VariablesStorage,
                agents_amount = 25,
                generate_new = False,
                ) -> dict:

        #storage = variables_storage.VariablesStorage(scenario_amount)

        if generate_new:
            d_target = [[random.uniform(0,4) for j in range(scenario_amount)] for i in range(agents_amount)]
            g_res = [[random.uniform(0,3) for j in range(scenario_amount)] for i in range(agents_amount)]
            storage.update(d_target, g_res)
        

