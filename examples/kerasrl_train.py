import warnings
warnings.filterwarnings("ignore")
import numpy as np
import gym
import gym_lawnmower
from keras import models, layers, optimizers
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from kerasrl_extensions import *
import os
import datetime


INPUT_SHAPE = (9, 9)
WINDOW_LENGTH = 4
STEPS = 1750000
STEPS = 5000000
STEPS = 7500000
STEPS = 10000000

datetime_string = datetime.datetime.now().strftime("%Y%m%d-%H%M")


class LawnmowerProcessor(Processor):
    """
    Transforms observations, state-batches and rewards of
    Flappy-Bird.
    """

    def process_observation(self, observation):
        """
        Takes an observation, resizes it and turns it into greyscale.
        """
        processed_observation = observation
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """
        Normalizes a batch of observations.
        """
        processed_batch = batch.astype('float32') / 6.
        return processed_batch

    def process_reward(self, reward):
        """
        Clips the rewards.
        """
        return reward


def main():

    # Get the environment and extract the number of actions.
    environment_name = "lawnmower-medium-obstacles-v0"
    environment = gym.make(environment_name)
    environment.print_description()
    nb_actions = environment.action_space.n

    # Build the model.
    model = build_model_cnn((WINDOW_LENGTH,) + INPUT_SHAPE, nb_actions)
    print(model.summary())

    # Create sequential memory for memory replay.
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    # Process environment inputs and outputs.
    processor = LawnmowerProcessor()

    # Use epsilon-greedy as our policy.
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=int(STEPS * 0.8))

    # Instantiate and compile our agent.
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)
    dqn.compile(optimizers.Adam(lr=.00025), metrics=['mae'])

    # Set up some callbacks for training.
    checkpoint_weights_filename = 'dqn_' + environment_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(environment_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [TensorboardCallback(os.path.join("tensorboard", datetime_string))]
    callbacks += [FileLogger(log_filename, interval=100)]

    # Train the agent.
    dqn.fit(environment, callbacks=callbacks, nb_steps=STEPS, log_interval=10000)

    # Save the final networkt after training.
    weights_filename = 'dqn_{}_weights.h5f'.format(environment_name)
    dqn.save_weights(weights_filename, overwrite=True)

    # Run the agent.
    dqn.test(environment, nb_episodes=10, visualize=False)

def build_model_cnn(input_shape, actions):
    """
    Creates a Convolutional Neural Network.
    """
    model = models.Sequential()
    model.add(layers.Permute((2, 3, 1), input_shape=input_shape))
    model.add(layers.Convolution2D(32, (4, 4), strides=(2, 2), activation="relu"))
    model.add(layers.Convolution2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(actions, activation="linear"))
    return model

def build_model_dense(input_shape, actions):
    """
    Creates a fully connected Neural Network.
    """
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(actions, activation="linear"))
    return model


if __name__ == "__main__":
    main()
