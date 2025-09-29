import irsim
from custom_behaviour import Agent

def record_metrics(robot_list): 
    for robot in robot_list:
        agent = Agent(robot)
        if "crash_stats" not in agent.metrics or not agent.metrics["crash_stats"].get("crashed", False):
            agent.update_metrics(robot)

episodes = 10

for e in range(episodes):
    env = irsim.make('conf.yaml', projection="2d") # initialize the environment with the configuration file
    env.load_behavior("custom_behaviour")

    # Reset metrics for all agents
    for robot in env.robot_list:
        agent = Agent(robot)
        agent.reset_metrics()

    # Run the episode
    for i in range(100 + e * 50): # run the simulation for 300 steps
        env.step()  # update the environment
        record_metrics(env.robot_list)
        env.render() # render the environment

        if env.done(): break # check if the simulation is done

    # Print stats
    first_crash = None
    num_crashes = 0
    cumulated_mse_per_frame = 0

    for robot in env.robot_list:
        agent = Agent(robot)
        mse = agent.read_mse()
        cumulated_mse_per_frame += mse / i if mse is not None else 0
        mse_per_frame = mse / i
        print(f"Episode {e}, Step {i}, Robot {robot.id}, MSE: {(mse_per_frame):.4f}")
        num_crashes += int(agent.read_crashed())
        
    first_crash = min([agent.read_crash_time() for robot in env.robot_list if (agent := Agent(robot)).read_crashed()] + [float('inf')]) # inf added in case of no crashes
    cumulated_mse_per_frame /= len(env.robot_list)
    print(f"Episode {e} ended after {i} steps with {num_crashes} crashes. First crash at {first_crash} seconds.")
    print(f"Average MSE for episode {e}: {(cumulated_mse_per_frame):.4f}")
    env.end()