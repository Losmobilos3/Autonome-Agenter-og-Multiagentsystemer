from Assignment_3_Grid.Simulation import Simulation
from Assignment_3_Grid.settings import SIMULATION_SIZE, N_AGENTS, N_FRUITS

simulation = Simulation(
    n_agents=N_AGENTS,
    n_fruits=N_FRUITS,
)

simulation.run()