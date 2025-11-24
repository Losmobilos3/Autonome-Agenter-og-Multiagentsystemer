from sim import Simulation
import matplotlib.pyplot as plt
from matplotlib import animation

sim = Simulation(
    no_agents= 1,
    no_fruits= 20,
    width = 70,
    height = 40,
)

sim.run_episodes(no_episodes=100, max_steps_per_episode=300)

sim.init_env()

scatter = sim.setup_plot()

print("START")

ani = animation.FuncAnimation(
    fig=sim.fig,
    func=sim.animate_frame,
    frames=1000,
    interval=60,
    blit=True,
    repeat=False
)

try:
    ani.save('simulation_animation.mp4', writer='ffmpeg', fps=60)
    print("Animation saved successfully")
except Exception as e:
    print(f"Error saving animation: {e}")

plt.close(sim.fig)