import time
import gym
import gym_lawnmower
import random

environment_ids = [
    "lawnmower-small-v0",
    "lawnmower-small-obstacles-v0",
    "lawnmower-medium-v0",
    "lawnmower-medium-obstacles-v0",
    "lawnmower-big-v0",
    "lawnmower-big-obstacles-v0"
]
environment_id = random.choice(environment_ids)
environment = gym.make(environment_id)
environment.print_description()

observation = environment.reset()
done = False
while done == False:
    action = environment.action_space.sample()
    observation, reward, done, info = environment.step(action)
    environment.render()
    print("Reward", reward, "Action", action, "Done", done)
    time.sleep(0.5)
print(info)
