from sim import Simulation
import matplotlib as plt
from matplotlib import animation

sim = Simulation(
    no_agents= 4,
    no_leaders = 1,
    width = 70,
    height = 40,
    view_distance = 20,
    protect_distance = 4
)

scatter = sim.setup_plot()

NO_frames = 2000

# for i in range(NO_frames):
#     sim.step()
#     sim.animate_frame(i)
#     if i % 10 == 0:
#         print(f"Completed frame {i+1}/{NO_frames}")

print("START")

ani = animation.FuncAnimation(
    fig=sim.fig,                 # The figure object
    func=sim.animate_frame,      # The function to call for each frame
    frames=2000,                  # Run for 300 steps
    interval=10,                 # Delay between frames in milliseconds (10 ms = 100 FPS)
    blit=True                    # Optimizes drawing by only redrawing changed elements
)

ani.save('simulation_animation.mp4', writer='ffmpeg', fps=60)

sim.plot_avg_distance_to_leader()
sim.plot_avg_distance_to_followers()
sim.plot_diff_distance_to_leader()
sim.plot_diff_distance_to_followers()