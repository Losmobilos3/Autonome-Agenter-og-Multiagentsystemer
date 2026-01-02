from Assignment_3_Grid.Simulation import Simulation
from Assignment_3_Grid.settings import SIMULATION_SIZE

simulation = Simulation(
    simulation_size=SIMULATION_SIZE,
    n_agents=1,
    n_fruits=5,
    steps=1000,
)

simulation.run()