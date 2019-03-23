from gym.envs.registration import register

kwargs = {
    "width": 4,
    "height": 4,
    "obstacle_number": 0,
    "max_steps": 50,
}
register(
    id="lawnmower-small-v0",
    entry_point='gym_lawnmower.envs:LawnmowerEnv',
    kwargs=kwargs
)

kwargs = {
    "width": 4,
    "height": 4,
    "obstacle_number": 2,
    "max_steps": 50,
}
register(
    id="lawnmower-small-obstacles-v0",
    entry_point='gym_lawnmower.envs:LawnmowerEnv',
    kwargs=kwargs
)

kwargs = {
    "width": 8,
    "height": 8,
    "obstacle_number": 0,
    "max_steps": 150,
}
register(
    id="lawnmower-medium-v0",
    entry_point='gym_lawnmower.envs:LawnmowerEnv',
    kwargs=kwargs
)

kwargs = {
    "width": 8,
    "height": 8,
    "obstacle_number": 4,
    "max_steps": 150,
}
register(
    id="lawnmower-medium-obstacles-v0",
    entry_point='gym_lawnmower.envs:LawnmowerEnv',
    kwargs=kwargs
)

kwargs = {
    "width": 16,
    "height": 16,
    "obstacle_number": 0,
    "max_steps": 750,
}
register(
    id="lawnmower-big-v0",
    entry_point='gym_lawnmower.envs:LawnmowerEnv',
    kwargs=kwargs
)

kwargs = {
    "width": 16,
    "height": 16,
    "obstacle_number": 8,
    "max_steps": 750,
}
register(
    id="lawnmower-big-obstacles-v0",
    entry_point='gym_lawnmower.envs:LawnmowerEnv',
    kwargs=kwargs
)
