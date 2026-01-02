from Assignment_3_Grid.Simulation import Simulation
from Assignment_3_Grid.settings import SIMULATION_SIZE, N_AGENTS, N_FRUITS

simulation = Simulation(
    simulation_size=SIMULATION_SIZE,
    n_agents=N_AGENTS,
    n_fruits=N_FRUITS,
    steps=1000,
)

simulation.run()