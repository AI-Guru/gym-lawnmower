from gym.envs.registration import register

# A very small map. No obstacles.
kwargs = {
    "width": 5,
    "height": 5,
    "obstacle_number": 0,
    "max_steps": 50,
}
register(
    id="lawnmower-small-v0",
    entry_point='gym_lawnmower.envs:LawnmowerEnv',
    kwargs=kwargs
)

# A very small map. Some random obstacles.
kwargs = {
    "width": 5,
    "height": 5,
    "obstacle_number": 2,
    "max_steps": 50,
}
register(
    id="lawnmower-small-obstacles-v0",
    entry_point='gym_lawnmower.envs:LawnmowerEnv',
    kwargs=kwargs
)

# A medium sized map. No obstacles.
kwargs = {
    "width": 9,
    "height": 9,
    "obstacle_number": 0,
    "max_steps": 150,
}
register(
    id="lawnmower-medium-v0",
    entry_point='gym_lawnmower.envs:LawnmowerEnv',
    kwargs=kwargs
)

# A medium sized map. Some random obstacles.
kwargs = {
    "width": 9,
    "height": 9,
    "obstacle_number": 4,
    "max_steps": 150,
}
register(
    id="lawnmower-medium-obstacles-v0",
    entry_point='gym_lawnmower.envs:LawnmowerEnv',
    kwargs=kwargs
)

# A very big map. No obstacles.
kwargs = {
    "width": 17,
    "height": 17,
    "obstacle_number": 0,
    "max_steps": 750,
}
register(
    id="lawnmower-big-v0",
    entry_point='gym_lawnmower.envs:LawnmowerEnv',
    kwargs=kwargs
)

# A very big map. Some random obstacles.
kwargs = {
    "width": 17,
    "height": 17,
    "obstacle_number": 8,
    "max_steps": 750,
}
register(
    id="lawnmower-big-obstacles-v0",
    entry_point='gym_lawnmower.envs:LawnmowerEnv',
    kwargs=kwargs
)
