import irsim
from custom_behaviour import Agent

episodes = 10

for e in range(episodes):
    env = irsim.make('conf.yaml', projection="2d") # initialize the environment with the configuration file
    env.load_behavior("custom_behaviour")
    for i in range(100 + e * 50): # run the simulation for 300 steps
        env.step()  # update the environment
        env.render() # render the environment

        if env.done(): break # check if the simulation is done

    # Print stats
    first_crash = None
    num_crashes = 0
    for robot in env.robot_list:
        agent = Agent(robot)
        mse = agent.read_mse()
        print(f"Episode {e}, Step {i}, Robot {robot.id}, MSE: {mse:.4f}")
        num_crashes += int(agent.read_crashed())
        first_crash = agent.read_crash_time() if first_crash is None else first_crash

    print(f"Episode {e} ended after {i} steps with {num_crashes} crashes. First crash at {first_crash} seconds.")
    env.end()

env.end() # close the environment