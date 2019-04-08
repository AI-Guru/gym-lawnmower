# Lawnmower environment for OpenAI Gym.

TODO how to Run

- train, test, tensorboard

TODO picture

TODO explain rewards

TODO examples

'''
import time
import gym
import gym_lawnmower
import random

environment = gym.make("lawnmower-medium-obstacles-v0")
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

'''

## Available environments.

These environments are available. Their names are very speaking.
- "lawnmower-small-v0")
- "lawnmower-small-obstacles-v0")
- "lawnmower-medium-v0")
- "lawnmower-medium-obstacles-v0")
- "lawnmower-big-v0")
- "lawnmower-big-obstacles-v0")
