from sim import Simulation
import matplotlib as plt
from matplotlib import animation

sim = Simulation(
    no_agents= 4,
    no_leaders = 1,
    width = 70,
    height = 40,
    view_distance = 15,
    protect_distance = 4
)

scatter = sim.setup_plot()

print("START")

ani = animation.FuncAnimation(
    fig=sim.fig,
    func=sim.animate_frame,
    frames=2000,
    interval=10,
    blit=True
)

ani.save('simulation_animation.mp4', writer='ffmpeg', fps=60)

sim.plot_avg_distance_to_leader()
sim.plot_avg_distance_to_followers()
sim.plot_diff_distance_to_leader()
sim.plot_diff_distance_to_followers()