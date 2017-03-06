#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model
from keras.optimizers import Adam

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import AtariPreprocessor
from deeprl_hw2.core import *
from deeprl_hw2.policy import *

import gym

def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """

    input_shape = (None, input_shape[0], input_shape[1], window)

    input_shape = (80, 80, 4)
    num_actions = 6
    state = Input(shape=input_shape)
    # First convolutional layer
    x = Convolution2D(16, 8, 8, border_mode='valid', activation='relu')(state)
    # Second convolutional layer
    x = Convolution2D(32, 4, 4, border_mode='valid', activation='relu')(x)
    # flatten the tensor
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    # output layer
    y_pred = Dense(num_actions)(x)

    model = Model(input=state, output=y_pred)

    return model


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument('--window', default=4, help='how many frames are used each time')
    parser.add_argument('--new_size', default=(80, 80), help='new size')
    parser.add_argument('--batch_size', default=32, help='Batch size')
    parser.add_argument('--replay_buffer_size', default=10000, help='Replay buffer size')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    parser.add_argument('--alpha', default=0.0001, help='Learning rate')
    parser.add_argument('--epsilon', default=0.05, help='Exploration probability for epsilon-greedy')
    parser.add_argument('--target_update_freq', default=0.05, help='Exploration probability for epsilon-greedy')
    parser.add_argument('--num_burn_in', default=0.05, help='Exploration probability for epsilon-greedy')
    parser.add_argument('--num_iterations', default=1000, help='Exploration probability for epsilon-greedy')
    parser.add_argument('--max_episode_length', default=100, help='Exploration probability for epsilon-greedy')
    parser.add_argument('--train_freq', default=0.05, help='Exploration probability for epsilon-greedy')
    parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    # args.input_shape = tuple(args.input_shape)

    # args.output = get_output_folder(args.output, args.env)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

    # keras model
    env = gym.make('SpaceInvaders-v0')
    num_actions = env.action_space.n

    preprocessor = AtariPreprocessor(args.new_size)
    q_network = create_model(args.window, args.new_size, num_actions)
    memory = ReplayMemory(args.replay_buffer_size, args.window)
    policy = LinearDecayGreedyEpsilonPolicy(args.epsilon, 0, 100)
    sess = tf.Session()
    dqn_agent = DQNAgent(q_network, preprocessor, memory, policy, args.gamma, \
             args.target_update_freq, args.num_burn_in, args.train_freq, args.batch_size, sess)

    dqn_agent.fit(env, args.num_iterations, args.max_episode_length)

    # while 1:
    #     env = gym.make('SpaceInvaders-v0')
    #     action = 5
    #     nextstate, reward, is_terminal, debug_info = env.step(action)
    #     while not is_terminal:
    #         nextstate, reward, is_terminal, debug_info = env.step(action)
    #         env.render()



if __name__ == '__main__':
    main()
