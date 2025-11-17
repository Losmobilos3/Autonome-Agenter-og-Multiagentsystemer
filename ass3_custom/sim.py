import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

import numpy as np
import os
from agent import Agent, Leader, Follower

class Simulation:
    def __init__(self, no_agents, no_leaders, width, height, view_distance, protect_distance, screen_ratio=0.9):
        self.size = np.vstack([width, height])
        self.screen_ratio = screen_ratio
        self.agents: list[Agent] = []
        self.avg_dist_leader = [] 
        self.avg_dist_followers = [] 

        # Initiate performance metrics
        avg_sep_metric = []
        avg_align_metric = []
        avg_coh_metric = []

        # Append followers to the agent list
        for n in range(no_agents):
            self.agents.append(Follower(view_distance, protect_distance, self))
            
        # Append leader(s) to the agent list
        for n in range(no_leaders):
            self.agents.append(Leader(view_distance, protect_distance, self))

        self.fig, self.ax = plt.subplots(figsize=(20, 15))


    def step(self):
        for agent in self.agents:
            agent.act()
            agent.update()

    def setup_plot(self):
        self.ax.axis("equal")
        self.ax.set_xlim(0, self.size[0])
        self.ax.set_ylim(0, self.size[1])
        
        x_coords = [a.pos[0][0] for a in self.agents]
        y_coords = [a.pos[1][0] for a in self.agents]

        # init scatter plot (as before)
        self.scatter = self.ax.scatter(x_coords, y_coords, s=15, color=["blue" if isinstance(a, Follower) else "red" for a in self.agents])
        
        # Initialize a list for FOV patches
        self.fov_patches = []
        
        for a in self.agents:
            wedge = Wedge(
                center=(a.pos[0][0], a.pos[1][0]), 
                r=a.view_distance, 
                theta1=0,
                theta2=360,
                alpha=0.1,  # Transparency
                color="grey" if isinstance(a, Follower) else "red", 
                animated=True # Mark as animated for better performance with blitting
            )
            self.ax.add_patch(wedge)
            self.fov_patches.append(wedge)
        
        # Return the new artists along with the old scatter for the animation
        return self.scatter, *self.fov_patches  # Use tuple unpacking to return all artists

    def save_metrics(self):
        self.avg_dist_leader.append(self.avg_dist_to_leader())
        self.avg_dist_followers.append(self.avg_dist_to_followers())
        
        
    def avg_dist_to_leader(self):
        leaders = [a for a in self.agents if isinstance(a, Leader)]
        followers = [a for a in self.agents if isinstance(a, Follower)]

        distances = []
        for f in followers:
            for l in leaders:
                distances.append(np.linalg.norm(f.pos - l.pos))
            # dists = [np.linalg.norm(f.pos - l.pos) for l in leaders]
            # distances.append(dists) 
                        
        return np.mean(distances)
    
    def avg_dist_to_followers(self):
        followers = [a for a in self.agents if isinstance(a, Follower)]

        distances = []
        done_followers = []
        for f in followers:
            for other in followers:
                if other != f and other not in done_followers:
                    distances.append(np.linalg.norm(f.pos - other.pos))
            done_followers.append(f)
        
        return np.mean(distances)

    def plot_avg_distance_to_leader(self, filename="avg_distance_to_leader.png", folder="plots", fps=60):
        os.makedirs(folder, exist_ok=True)
        
        # Convert frames to seconds
        seconds = np.arange(len(self.avg_dist_leader)) / fps
        
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(seconds, self.avg_dist_leader, label='Avg Distance to Leader')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Average Distance')
        ax.set_title('Average Distance of Followers to Leader')
        ax.legend()
        ax.grid(True)
        
        filepath = os.path.join(folder, filename)
        fig.savefig(filepath)
        plt.close(fig)
        
        
    def plot_avg_distance_to_followers(self, filename="avg_distance_to_followers.png", folder="plots", fps=60):
        os.makedirs(folder, exist_ok=True)
        
        # Convert frames to seconds
        seconds = np.arange(len(self.avg_dist_followers)) / fps
        
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(seconds, self.avg_dist_followers, label='Avg Distance to Followers')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Average Distance')
        ax.set_title('Average Between Distance of Followers')
        ax.legend()
        ax.grid(True)

        filepath = os.path.join(folder, filename)
        fig.savefig(filepath)
        plt.close(fig)

    def plot_diff_distance_to_leader(self, filename="diff_distance_to_leader.png", folder="plots", fps=60, window_size=60):
        os.makedirs(folder, exist_ok=True)
        
        # Convert frames to seconds
        seconds = np.arange(len(self.avg_dist_leader)) / fps
        
        # Caluclate instantaneoues diff between frames
        diff = np.diff(self.avg_dist_leader, prepend=0)
        diff[0] = 0  
        
        # Apply moving average to smooth the data over a time window
        # window_size is in frames (default 60 frames = 1 second at 60 FPS)
        # Use moving average for smoothing, 60 frame window equal to 1 second
        smoothed_diff = np.convolve(diff, np.ones(window_size)/window_size, mode='same')
        
        fig, ax = plt.subplots(figsize=(10,6))
        # ax.plot(seconds, diff, label='Instantaneous Rate of Change', alpha=0.3, linewidth=0.5)
        ax.plot(seconds, smoothed_diff, label=f'Smoothed rate of change ({window_size/fps:.1f}s window)', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Change in Distance')
        ax.set_title('Rate of Change in Average Distance to Leader')
        ax.legend()
        ax.grid(True)

        filepath = os.path.join(folder, filename)
        fig.savefig(filepath)
        plt.close(fig)
  
    def plot_diff_distance_to_followers(self, filename="diff_distance_to_followers.png", folder="plots", fps=60, window_size=60):
        os.makedirs(folder, exist_ok=True)
        
        # Convert frames to seconds
        seconds = np.arange(len(self.avg_dist_followers)) / fps
        
        # Caluclate instantaneoues diff between frames
        diff = np.diff(self.avg_dist_followers, prepend=0)
        diff[0] = 0  
        
        # Apply moving average to smooth the data over a time window
        # window_size is in frames (default 60 frames = 1 second at 60 FPS)
        # Use moving average for smoothing, 60 frame window equal to 1 second
        smoothed_diff = np.convolve(diff, np.ones(window_size)/window_size, mode='same')
        
        fig, ax = plt.subplots(figsize=(10,6))
        # ax.plot(seconds, diff, label='Instantaneous Rate of Change', alpha=0.3, linewidth=0.5)
        ax.plot(seconds, smoothed_diff, label=f'Smoothed rate of change ({window_size/fps:.1f}s window)', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Change in Distance')
        ax.set_title('Rate of Change in Average Distance Between Followers')
        ax.legend()
        ax.grid(True)

        filepath = os.path.join(folder, filename)
        fig.savefig(filepath)
        plt.close(fig)

    def animate_frame(self, i):
        if i % 100 == 0:
            print(f"Animating frame {i}")

        # Update the simulation state
        self.step() 
        
        # Extract agent positions
        x_coords = [a.pos[0][0] for a in self.agents]
        y_coords = [a.pos[1][0] for a in self.agents]
        
        # Update the data in the existing scatter plot object
        self.scatter.set_offsets(np.c_[x_coords, y_coords])

        # Update each FOV patch
        updated_patches = []
        for a, patch in zip(self.agents, self.fov_patches):
            # Update center position
            patch.set_center((a.pos[0][0], a.pos[1][0])) 
            updated_patches.append(patch)

        # Save performance metrics
        self.save_metrics()
        
        # Return all updated plot objects (scatter and patches)
        return self.scatter, *updated_patches