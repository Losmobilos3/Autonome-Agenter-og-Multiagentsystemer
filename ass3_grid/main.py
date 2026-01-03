from ass3_grid.config import FRAMES
from sim import Simulation
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

no_agent_runs = [1, 2, 3, 4, 5]

for no_agents in no_agent_runs:
    print(f"START TRAINING RUN {no_agents}")

    with open("MARL_data.txt", "a") as f:
        f.write(f"Starting run with {no_agents} agents. (performance, fruits_level_1, fruits_level_2)\n")

    sim = Simulation(
        no_agents= no_agents,
        no_fruits= 30,
        width = 35,
        height = 20,
    )

    sim.run_episodes(no_episodes=100, max_steps_per_episode=300)

    sim.init_env()

    sim.setup_plot()

    print(f"START ANIMATION SAVE FOR RUN {no_agents}")

    ani = animation.FuncAnimation(
        fig=sim.fig,
        func=sim.animate_frame,
        frames=FRAMES,
        interval=1,
        blit=True,
        repeat=False
    )

    try:
        ani.save(f'MARL_simulation_animation_{no_agents}.mp4', writer='ffmpeg', fps=24)
        print("Animation saved successfully")
    except Exception as e:
        print(f"Error saving animation: {e}")

    plt.close(sim.fig)