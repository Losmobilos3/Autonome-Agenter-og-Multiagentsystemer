import irsim
import numpy as np

env = irsim.make()
env.load_behavior("follower_behaviour")
# for robot in env.robot_list:
    # robot.set_velocity = np.random.rand(2)

for _i in range(10000):
    env.step()
    env.render(figure_kwargs={"dpi": 100})
env.end()