import irsim

env = irsim.make()
env.load_behavior("custom_behaviour")

for _i in range(1000):
    env.step()
    env.render(figure_kwargs={"dpi": 100})
env.end()