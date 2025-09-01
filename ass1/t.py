#import custom_behaviour
import irsim

env = irsim.make('conf.yaml', projection="2d") # initialize the environment with the configuration file
env.load_behavior("custom_behaviour")

for i in range(300): # run the simulation for 300 steps
    if i == 60:
        print("hej")
        env.robot_list[0].set_goal([1, 1, 0])

    env.step()  # update the environment
    env.render() # render the environment

    if env.done(): break # check if the simulation is done
        
env.end() # close the environment