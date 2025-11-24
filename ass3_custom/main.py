from sim import Simulation
import matplotlib as plt
from matplotlib import animation

sim = Simulation(
    no_agents= 1,
    no_fruits= 2,
    width = 70,
    height = 40,
)

scatter = sim.setup_plot()

print("START")

ani = animation.FuncAnimation(
    fig=sim.fig,
    func=sim.animate_frame,
    frames=1000,
    interval=60,
    blit=True
)

ani.save('simulation_animation.mp4', writer='ffmpeg', fps=60)