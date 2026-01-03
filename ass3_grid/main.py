from sim import Simulation
import matplotlib.pyplot as plt
from matplotlib import animation

sim = Simulation(
    no_agents= 2,
    no_fruits= 20,
    width = 35,
    height = 20,
)

sim.run_episodes(no_episodes=50, max_steps_per_episode=300)

sim.init_env()

sim.setup_plot()

print("START")

ani = animation.FuncAnimation(
    fig=sim.fig,
    func=sim.animate_frame,
    frames=500,
    interval=200,
    blit=True,
    repeat=False
)

try:
    ani.save('simulation_animation.mp4', writer='ffmpeg', fps=60)
    print("Animation saved successfully")
except Exception as e:
    print(f"Error saving animation: {e}")

plt.close(sim.fig)


# https://chatgpt.com/c/6957adcd-f698-8326-98fa-e23f44ea6b35