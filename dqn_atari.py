#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input, Permute)


from keras.models import Model

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import AtariPreprocessor
from deeprl_hw2.core import *
from deeprl_hw2.policy import *

import gym


def create_model_deep(window, input_shape, num_actions,
                      model_name='q_network_deep'):  # noqa: D103
    """Create the Deep-Q-network model.

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
    model = None
    if model_name is "q_network_deep" or "q_network_double":

        input_shape = (input_shape[0], input_shape[1], window)

        state = Input(shape=input_shape)
        # First convolutional layer
        x = Convolution2D(16, 8, 8, border_mode='valid')(state)
        x = Activation('relu')(x)
        # Second convolutional layer
        x = Convolution2D(32, 4, 4, border_mode='valid')(x)
        x = Activation('relu')(x)
        # flatten the tensor
        x = Flatten()(x)
        x = Dense(256)(x)
        # output layer
        y_pred = Dense(num_actions)(x)

        model = Model(input=state, output=y_pred)

    elif model_name is "q_network_duel":
        input_shape = (input_shape[0], input_shape[1], window)

        state = Input(shape=input_shape)
        # conv1
        x = Convolution2D(16, 8, 8, border_mode='valid')(state)
        x = Activation('relu')(x)
        # conv2
        x = Convolution2D(32, 4, 4, border_mode='valid')(x)
        x = Activation('relu')(x)

        x = Flatten()(x)
        # value output
        x_val = Dense(256)(x)
        y_val = Dense(1)(x_val)

        # advantage output
        x_advantage = Dense(256)(x)
        y_advantage = Dense(num_actions)(x_advantage)
        # mean advantage
        y_advantage_mean = tf.reduce_mean(y_advantage, axis=1)

        y_q = y_val + y_advantage - y_advantage_mean

        model = Model(input=state, output=y_q)

    elif model_name is "q_network_linear":
        input_shape = (input_shape[0], input_shape[1], window)
        state = Input(shape=input_shape)
        x = Flatten()(state)
        x = Dense(256)(x)

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
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--window', default=4, help='how many frames are used each time')
    parser.add_argument('--new_size', default=(80, 80), help='new size')
    parser.add_argument('--batch_size', default=32, help='Batch size')
    parser.add_argument('--replay_buffer_size', default=1000000, help='Replay buffer size')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    parser.add_argument('--alpha', default=0.0001, help='Learning rate')
    parser.add_argument('--epsilon', default=0.05, help='Exploration probability for epsilon-greedy')
    parser.add_argument('--target_update_freq', default=50, help='Exploration probability for epsilon-greedy')
    parser.add_argument('--num_burn_in', default=50, help='Exploration probability for epsilon-greedy')
    parser.add_argument('--num_iterations', default=100000, help='Exploration probability for epsilon-greedy')
    parser.add_argument('--max_episode_length', default=300, help='Exploration probability for epsilon-greedy')
    parser.add_argument('--train_freq', default=500, help='Exploration probability for epsilon-greedy')
    parser.add_argument('--experience_replay', default=False, help='Choose whether or not to use experience replay')
    parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--save_freq', default=10000, type=int, help='model save frequency')
    args = parser.parse_args()
    # args.input_shape = tuple(args.input_shape)

    # args.output = get_output_folder(args.output, args.env)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

    # keras model

    output_folder = get_output_folder("./result/", args.env)

    env = gym.make(args.env)
    num_actions = env.action_space.n

    preprocessor = AtariPreprocessor(args.new_size)

    q_network_online = create_model_deep(args.window, args.new_size, num_actions, "q_network_double")
    q_network_target = create_model_deep(args.window, args.new_size, num_actions, "q_network_double")

    memory = ReplayMemory(args.replay_buffer_size, args.window)
    policy = LinearDecayGreedyEpsilonPolicy(args.epsilon, 0, 1000)
    with tf.Session() as sess:
        dqn_agent = DQNAgent((q_network_online, q_network_target), preprocessor, memory, policy, args.gamma, \
                             args.target_update_freq, args.num_burn_in, args.train_freq, args.batch_size, \
                             args.experience_replay, sess)

        optimizer = tf.train.AdamOptimizer(learning_rate=args.alpha)
        dqn_agent.compile(optimizer, mean_huber_loss)
        dqn_agent.fit(env, args.num_iterations, output_folder, args.save_freq, args.max_episode_length)
        # dqn_agent.evaluate(env, 10)

    # while 1:
    #     env = gym.make('SpaceInvaders-v0')
    #     action = 5
    #     nextstate, reward, is_terminal, debug_info = env.step(action)
    #     while not is_terminal:
    #         nextstate, reward, is_terminal, debug_info = env.step(action)
    #         env.render()


if __name__ == '__main__':
    main()
