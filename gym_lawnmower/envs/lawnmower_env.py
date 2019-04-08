import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import numpy as np
import random


# Indices for grid.
GRID_INDEX_GRASS = 0
GRID_INDEX_MOWED = 1
GRID_INDEX_OBSTACLE = 2
GRID_INDEX_MOWER_UP = 3
GRID_INDEX_MOWER_RIGHT = 4
GRID_INDEX_MOWER_DOWN = 5
GRID_INDEX_MOWER_LEFT = 6
GRID_INDICES_MOWER = [GRID_INDEX_MOWER_UP, GRID_INDEX_MOWER_RIGHT, GRID_INDEX_MOWER_DOWN, GRID_INDEX_MOWER_LEFT]
GRID_INDICES = range(0, 7)
GRID_INDEX_MAX = 6

# Indices for actions.
ACTION_INDEX_FORWARD = 0
ACTION_INDEX_LEFT = 1
ACTION_INDEX_RIGHT = 2
ACTION_INDICES = [ACTION_INDEX_FORWARD, ACTION_INDEX_LEFT, ACTION_INDEX_RIGHT]
ACTION_INDEX_MAX = max(ACTION_INDICES)

# Orientations.
ORIENTATION_INDEX_UP = 0
ORIENTATION_INDEX_RIGHT = 1
ORIENTATION_INDEX_DOWN = 2
ORIENTATION_INDEX_LEFT = 3
ORIENTATION_INDICES = [ORIENTATION_INDEX_UP, ORIENTATION_INDEX_RIGHT, ORIENTATION_INDEX_DOWN, ORIENTATION_INDEX_LEFT]
ORIENTATION_INDEX_MAX = max(ORIENTATION_INDICES)

# Mapping orientations to grid.
orientation_to_grid = {
    ORIENTATION_INDEX_UP: GRID_INDEX_MOWER_UP,
    ORIENTATION_INDEX_RIGHT: GRID_INDEX_MOWER_RIGHT,
    ORIENTATION_INDEX_DOWN: GRID_INDEX_MOWER_DOWN,
    ORIENTATION_INDEX_LEFT: GRID_INDEX_MOWER_LEFT,
}

# Mapping grid to characters.
grid_to_character = {
    GRID_INDEX_GRASS: ".",
    GRID_INDEX_MOWED: " ",
    GRID_INDEX_OBSTACLE: "X",
    GRID_INDEX_MOWER_UP: "^",
    GRID_INDEX_MOWER_RIGHT: ">",
    GRID_INDEX_MOWER_DOWN: "v",
    GRID_INDEX_MOWER_LEFT: "<",
}

TILE_SIZE = 64

def load_image(filename):
    root_path = os.path.abspath(os.path.dirname(__file__))
    image_path = os.path.join(root_path, "resources", filename)
    image = pygame.image.load(image_path)
    image = pygame.transform.scale(image, [TILE_SIZE, TILE_SIZE])
    return image

# Mapping grid to image.
grid_to_image = {
    GRID_INDEX_GRASS: load_image("Grass.png"),
    GRID_INDEX_MOWED: load_image("Mowed.png"),
    GRID_INDEX_OBSTACLE: load_image("Obstacle.png"),
    GRID_INDEX_MOWER_UP: load_image("MowerUp.png"),
    GRID_INDEX_MOWER_RIGHT: load_image("MowerRight.png"),
    GRID_INDEX_MOWER_DOWN: load_image("MowerDown.png"),
    GRID_INDEX_MOWER_LEFT: load_image("MowerLeft.png"),
}

# Rewards.
REWARD_EXCEEDED = -100
REWARD_STEP = -1
REWARD_NOT_MOWED = -1
REWARD_TOO_MANY_TURNS = -10
REWARD_MOWED = 10
REWARD_ALL_MOWED = 100
REWARD_OBSTACLE_COLLISION = -100

class LawnmowerEnv(gym.Env):
    """
    The lawnmower environment is a rectangular grid. The size is customizable.
    The border of the grid consists of obstacles. On  top of that other cells
    can be obstacles too. Everything else is mostly grass. If the robot steps
    on a cell with grass, the grass is mowed. If the robot hits an obstacle, the
    robot is damaged beyond repair.
    """

    metadata = {"render.modes": ["window", "console"]}

    def __init__(self, width, height, obstacle_number, max_steps):

        # Parameters.
        self.width = width
        self.height = height
        self.obstacle_number = obstacle_number
        self.max_steps = max_steps

        # Observation space.
        self.observation_space = spaces.Box(low=0, high=GRID_INDEX_MAX, shape=(self.width, self.height), dtype=np.uint8), # Target image.

        # Action space.
        self.action_space = spaces.Discrete(ACTION_INDEX_MAX + 1)

        # Prepare for pygame.
        self._pygame_screen = None

    def print_description(self):
        """
        Prints a human-readble description of the environment.
        """

        description = ""
        description += "Lawnmower environment." + "\n"
        description += "  Id:        {}".format(self.unwrapped.spec.id) + "\n"
        description += "  Grid size: {}, {}".format(self.width, self.height) + "\n"
        description += "  Obstacles: {}".format(self.obstacle_number) + "\n"
        description += "  Max-steps: {}".format(self.max_steps) + "\n"
        print(description)

    def step(self, action):
        """
        Performs one step of the simulation.
        """
        # The environment should be reset at the beginning.
        assert self._reset == True, "Did you reset the environment?"

        # Count the steps.
        self.current_step += 1

        # Terminate the simulation if the number of steps has been exceeded.
        if self.current_step == self.max_steps:
            observation = self._get_observation()
            reward = REWARD_EXCEEDED
            done = True
            info = { "done_reason" : "Steps exceeded." }

        # Do a normal simulation step.
        else :
            reward, done, info = self._perform_action(action)
            observation = self._get_observation()

        # Done for now.
        return observation, reward, done, info

    def _perform_action(self, action):
        """
        Performs a single action.
        """

        info = {}

        # Move forward and mow the lawn.
        if action == ACTION_INDEX_FORWARD:
            # Store the old position for later.
            old_x, old_y = self.mower_position

            # Compute the potential new position.
            new_x, new_y = self.mower_position
            if self.mower_orientation == ORIENTATION_INDEX_UP:
                new_y -= 1
            elif self.mower_orientation == ORIENTATION_INDEX_RIGHT:
                new_x += 1
            elif self.mower_orientation == ORIENTATION_INDEX_DOWN:
                new_y += 1
            elif self.mower_orientation == ORIENTATION_INDEX_LEFT:
                new_x -= 1

            # See if the move is possible.
            if self.grid[new_x, new_y] in [GRID_INDEX_GRASS, GRID_INDEX_MOWED]:
                # Mow the lawn and move the robot.
                new_was_grass = self.grid[new_x, new_y] == GRID_INDEX_GRASS
                self.grid[old_x, old_y] = GRID_INDEX_MOWED
                self.grid[new_x, new_y] = orientation_to_grid[self.mower_orientation]
                self.mower_position = (new_x, new_y)

                # Count unmowed grass on the whole grid.
                zero_count = self.width * self.height - np.count_nonzero(self.grid)
                all_mowed = zero_count == 0

                # The whole lawn has been mowed. Terminate and reward.
                if new_was_grass and all_mowed:
                    reward = REWARD_ALL_MOWED
                    done = True
                    info["done_reason"] = "All mowed"

                # A single patch of grass was mowed.
                elif new_was_grass:
                    reward = REWARD_MOWED
                    done = False

                # Nothing happened.
                else:
                    reward = REWARD_NOT_MOWED
                    done = False

            # Hitting an obstacle. Terminate and reward.
            else:
                reward = REWARD_OBSTACLE_COLLISION
                done = True
                info["done_reason"] = "Collision with obstacle."
                return reward, done, info

            # For counting turns. This is not a turn.
            self.turn_count = 0

        # Turn left. If there are too many turns, yield a negative reward.
        elif action == ACTION_INDEX_LEFT:
            self.mower_orientation = (self.mower_orientation - 1) % 4
            self.grid[self.mower_position] = orientation_to_grid[self.mower_orientation]
            reward = REWARD_NOT_MOWED_SAME_CELL
            done = False

            self.turn_count += 1
            if self.turn_count > 3:
                reward = REWARD_TOO_MANY_TURNS

        # Turn  right.
        elif action == ACTION_INDEX_RIGHT:
            self.mower_orientation = (self.mower_orientation + 1) % 4
            self.grid[self.mower_position] = orientation_to_grid[self.mower_orientation]
            reward = REWARD_NOT_MOWED_SAME_CELL
            done = False

            self.turn_count += 1
            if self.turn_count > 3:
                reward = REWARD_TOO_MANY_TURNS

        # You are not supposed to be here.
        else:
            assert False, str(self.mower_orientation)

        return reward, done, info


    def reset(self):
        """
        Resets the entire environment. Should be called in the beginning and
        everytime a simulation is over.
        """

        self._reset = True

        # Count the current step.
        self.current_step = 0

        # Count turns.
        self.turn_count = 0

        # Create an empty grid.
        self.grid = np.zeros((self.width, self.height)).astype("uint8")

        # Add mower.
        x, y = self._random_position()
        self.mower_position = (x, y)
        self.mower_orientation = random.randint(0, ORIENTATION_INDEX_MAX)
        self.grid[x, y] = orientation_to_grid[self.mower_orientation]

        # Add border.
        for x in [0, self.width - 1]:
            for y in range(0, self.height):
                self.grid[x, y] = GRID_INDEX_OBSTACLE
        for y in [0, self.height - 1]:
            for x in range(0, self.width):
                self.grid[x, y] = GRID_INDEX_OBSTACLE

        # Add obstacles.
        counter = 0
        while counter < self.obstacle_number:
            x, y = self._random_position()
            if self.grid[x, y] == GRID_INDEX_GRASS:
                self.grid[x, y] = GRID_INDEX_OBSTACLE
                counter += 1

        # Return the observation.
        observation = self._get_observation()
        return observation

    def render(self, mode="window", close=False):
        """
        Renders the grid.
        """
        assert self._reset == True, "Did you reset the environment?"

        # Renders the grid to the console.
        if mode == "console":
            grid_string = ""
            for y in range(self.height):
                for x in range(self.width):
                    assert self.grid[x, y] in GRID_INDICES, str(self.grid[x, y])
                    grid_character = grid_to_character[self.grid[x, y]]

                    grid_string += grid_character + " "
                grid_string += "\n"

            print(grid_string)

        # Renders the grid to a pygame window.
        elif mode == "window":
            # Lazy loading pygame.
            if self._pygame_screen == None:
                pygame.init()
                pygame.display.set_caption(self.unwrapped.spec.id)
                self.screen_width = self.width * TILE_SIZE
                self.screen_height = self.height * TILE_SIZE
                self._pygame_screen = pygame.display.set_mode((self.screen_width, self.screen_height))

            # Consume events.
            for event in pygame.event.get():
                pass

            # Render grid.
            for x, column in enumerate(self.grid):
                for y, cell in enumerate(column):
                    screen_x = x * TILE_SIZE
                    screen_y = y * TILE_SIZE
                    self._pygame_screen.blit(grid_to_image[cell], (screen_x, screen_y))

            # Flip
            pygame.display.flip()

    def _random_position(self):
        """
        Yields a random position on the grid that is not a border.
        """
        x = random.randint(1, self.width - 2)
        y = random.randint(1, self.height- 2)
        return x, y

    def _get_observation(self):
        """
        Yields the grid as an observation.
        """
        return self.grid.copy()
