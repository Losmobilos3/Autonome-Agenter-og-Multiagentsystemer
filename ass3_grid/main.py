from sim import Simulation
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

no_agent_runs = [1, 2, 3]

with open("SARL_data.txt", "w") as f:
    f.write("no_agent, performance, fruits_1, fruits_2\n")
    for no_agents in no_agent_runs:
        print(f"START TRAINING RUN {no_agents}")

        sim = Simulation(
            no_agents= no_agents,
            no_fruits= 30,
            width = 35,
            height = 20,
        )

        sim.run_episodes(no_episodes=50, max_steps_per_episode=300)

        sim.init_env()

        sim.setup_plot()

        print(f"START ANIMATION SAVE FOR RUN {no_agents}")

        frames = 500
        ani = animation.FuncAnimation(
            fig=sim.fig,
            func=sim.animate_frame,
            frames=frames,
            interval=1,
            blit=True,
            repeat=False
        )

        if sim.steps_used > 0:
            performance = sum(sim.total_fruits_collected) / sim.steps_used
        else:
            performance = sum(sim.total_fruits_collected) / frames

        f.write(f"{no_agents}, {performance}, {sim.total_fruits_collected[0]}, {sim.total_fruits_collected[1]}\n")

        try:
            ani.save(f'SARL_simulation_animation_{no_agents}.mp4', writer='ffmpeg', fps=24)
            print("Animation saved successfully")
        except Exception as e:
            print(f"Error saving animation: {e}")

        plt.close(sim.fig)