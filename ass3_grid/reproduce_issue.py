
from ass3_grid.sim import Simulation
import numpy as np
import torch

def run_test():
    sim = Simulation(no_agents=1, no_fruits=5, width=10, height=10)
    sim.init_env()
    
    agent = sim.agents[0]
    
    print("Initial Epsilon:", agent.epsilon)
    
    actions = []
    for i in range(100):
        sim.step(i)
        actions.append(agent.prior_action)
        
    print("Actions taken:", actions)
    print("Action counts:", {i: actions.count(i) for i in range(5)})
    print("Final Epsilon:", agent.epsilon)

if __name__ == "__main__":
    run_test()
