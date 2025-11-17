from sim import Simulation
import matplotlib as plt
from matplotlib import animation

sim = Simulation(
    no_agents= 4,
    no_fruits= 10,
    width = 70,
    height = 40,
)

scatter = sim.setup_plot()

print("START")

ani = animation.FuncAnimation(
    fig=sim.fig,
    func=sim.animate_frame,
    frames=10,
    interval=10,
    blit=True
)

ani.save('simulation_animation.mp4', writer='ffmpeg', fps=60)