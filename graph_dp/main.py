#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os

import keras
from keras import backend as K
from keras.layers import (Embedding, Dense, Dropout, Flatten, Input, Lambda)
from keras.models import model_from_json
from keras.models import Model

from deeprl.transition_dp.core.dqn import DQNAgent
from deeprl.transition_dp.core.objectives import mean_huber_loss
from deeprl.transition_dp.core.preprocessors import DependencyParsingPreprocessor
from deeprl.transition_dp.core.dp_envs import ArcEagerTransitionEnv
from deeprl.transition_dp.core.core import *
from deeprl.transition_dp.core.policy import *
from deeprl.transition_dp.core.utils import *

K.set_learning_phase(1)


def create_model(args, pretrained_embedding: np.ndarray, model_name='deep_q_network', trainable=True):
    """Create the Deep-Q-network model

    Parameters
    ----------
    args: Namespace
        model parameters
    pretrained_embedding: numpy.ndarray
        the pretrained embedding
    model_name: str
        Useful when debugging. Makes the model show up nicer in tensorboard.
    trainable: bool
        whether the model is trainable or not

    Returns
    -------
    keras.models.Model
      The Q-model.
    """

    state = Input(shape=(args.n_features,))
    model = None

    n, m = pretrained_embedding.shape
    print('shape', pretrained_embedding.shape)
    embedded = Embedding(n, m, embeddings_initializer=keras.initializers.constant(pretrained_embedding))(state)

    if model_name == "deep_q_network":
        print("Building " + model_name + " ...")

        # First convolutional layer

        x = Dense(args.hidden_size, activation=K.relu)(embedded)
        x = Dropout(args.dropout)(x)
        x = Flatten()(x)
        y_pred = Dense(args.n_actions, trainable=trainable)(x)

        model = Model(inputs=state, outputs=y_pred)

    elif model_name == "deep_q_network_double":
        print("Building " + model_name + " ...")

        x = Dense(args.hidden_size, activation=K.relu)(embedded)
        x = Dropout(args.dropout)(x)
        x = Flatten()(x)
        y_pred = Dense(args.n_actions, trainable=trainable)(x)

        model = Model(input=state, output=y_pred)

    elif model_name == "deep_q_network_duel":
        print("Building " + model_name + " ...")

        x = Dense(args.hidden_size, activation=K.relu)(embedded)
        x = Dropout(args.dropout)(x)
        x = Flatten()(x)

        y_pred = Dense(args.n_actions, trainable=trainable)(x)

        # value output
        x_val = Dense(args.hidden_size, trainable=trainable)(x)
        # x_val = Activation('relu')(x_val)
        y_val = Dense(1, trainable=trainable)(x_val)

        # advantage output
        x_advantage = Dense(args.hidden_size, trainable=trainable)(x)
        # x_advantage = Activation('relu')(x_advantage)
        y_advantage = Dense(args.n_actions, trainable=trainable)(x_advantage)
        # mean advantage
        y_advantage_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(y_advantage)

        y_q = Lambda(lambda x: x[0] + x[1] - x[2])([y_val, y_advantage, y_advantage_mean])

        model = Model(input=state, output=y_q)

    else:
        print("Model not supported")
        exit(1)

    return model


def get_output_folder(parent_dir, env_name):
    """Return save folder.

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
    parent_dir = parent_dir + '-run{}'.format(experiment_id) + '/'
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--network_name', default='deep_q_network', type=str, help='Type of model to use')
    parser.add_argument('--window', default=4, type=int, help='how many frames are used each time')
    parser.add_argument('--new_size', default=(84, 84), type=tuple, help='new size')
    parser.add_argument('--replay_buffer_size', default=750000, type=int, help='Replay buffer size')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--alpha', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--epsilon', default=0, type=float, help='Exploration probability for epsilon-greedy')
    parser.add_argument('--target_update_freq', default=10000, type=int,
                        help='Frequency for copying weights to target network')
    parser.add_argument('--num_burn_in', default=50000, type=int,
                        help='Number of prefilled samples in the replay buffer')
    parser.add_argument('--num_iterations', default=5000000, type=int,
                        help='Number of overal interactions to the environment')
    parser.add_argument('--max_episode_length', default=100, type=int, help='Terminate earlier for one episode')
    parser.add_argument('--train_freq', default=4, type=int, help='Frequency for training')
    parser.add_argument('--repetition_times', default=3, type=int, help='Parameter for action repetition')
    parser.add_argument('-o', '--output', default='atari-v0', type=str, help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--experience_replay', default=True, type=bool,
                        help='Choose whether or not to use experience replay')
    parser.add_argument('--train', default=True, type=bool, help='Train/Evaluate, set True if train the model')
    parser.add_argument('--max_grad', default=1.0, type=float, help='Parameter for huber loss')
    parser.add_argument('--model_num', default=5000000, type=int, help='specify saved model number during train')
    parser.add_argument('--log_dir', default='log', type=str, help='specify log folder to save evaluate result')
    parser.add_argument('--eval_num', default=100, type=int, help='number of evaluation to run')
    parser.add_argument('--save_freq', default=100000, type=int, help='model save frequency')

    parser.add_argument('--env', default='ArcEagerTransitionEnv-v0', help='DP env name')

    '''
    Files to be read into
    '''
    parser.add_argument('--data_path', default='../../data/', type=str, help='base path for all data files')
    parser.add_argument('--train_file', default='train.conll', type=str, help='training file')
    parser.add_argument('--dev_file', default='dev.conll', type=str, help='Development file')
    parser.add_argument('--test_file', default='test.conll', type=str, help='Test file')
    parser.add_argument('--embedding_file', default='en-cw.txt', type=str, help='Embedding file')
    parser.add_argument('--dependency_label_file', default='label_set.txt', type=str, help='DP Label file')
    parser.add_argument('--pos_tag', default='pos_set.txt', type=str, help='Pos Tagging file')
    parser.add_argument('--word_list', default='word_list.txt', type=str, help='Word List file')

    parser.add_argument('--n_features', default=48, type=int, help='Number of features')
    parser.add_argument('--n_actions', default=80, type=int, help='Number of actions')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate')
    parser.add_argument('--embedding_size', default=50, type=int, help='Embedding size')
    parser.add_argument('--hidden_size', default=200, type=int, help='Hidden size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--n_epochs', default=10, type=int, help='Number of Epochs')
    parser.add_argument('--batch_size', default=2048, type=int, help='Batch size')

    args = parser.parse_args()

    os.chdir(args.data_path)
    print("\nParameters:")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("")

    ''' Read Training and Evaluation Data '''
    tok_manager = TokManager(args)
    data_reader = ConllData(tok_manager)
    train_data = data_reader.read_conll(args.train_file)
    dev_data = data_reader.read_conll(args.dev_file)

    env = ArcEagerTransitionEnv(args.dependency_label_file)

    # define model object
    preprocessor = DependencyParsingPreprocessor(tok_manager)
    memory = ReplayMemory(args.replay_buffer_size)

    # Initiating policy for both tasks (training and evaluating)
    policy = LinearDecayGreedyEpsilonPolicy(args.epsilon, 0, 1000000)

    if not args.train:
        '''Evaluate the model'''
        # check model path
        if args.model_path is '':
            print("Model path must be set when evaluate")
            exit(1)

        # specific log file to save result
        log_file = os.path.join(args.log_dir, args.network_name, str(args.model_num))
        model_dir = os.path.join(args.model_path, args.network_name, str(args.model_num))

        with tf.Session() as sess:
            # load model
            with open(model_dir + ".json", 'r') as json_file:
                loaded_model_json = json_file.read()
                q_network_online = model_from_json(loaded_model_json)
                q_network_target = model_from_json(loaded_model_json)

            sess.run(tf.global_variables_initializer())

            # load weights into model
            q_network_online.load_weights(model_dir + ".h5")
            q_network_target.load_weights(model_dir + ".h5")

            dqn_agent = DQNAgent((q_network_online, q_network_target), preprocessor, memory, policy, args.n_actions,
                                 args.gamma, args.target_update_freq, args.num_burn_in, args.train_freq,
                                 args.batch_size, args.experience_replay, args.repetition_times, args.network_name,
                                 args.max_grad, args.env, sess)

            dqn_agent.evaluate(env, dev_data)
        exit(0)

    '''Train the model'''
    q_network_online = create_model(args, tok_manager.embeddings_matrix, args.network_name, True)
    q_network_target = create_model(args, tok_manager.embeddings_matrix, args.network_name, False)

    # create output dir, meant to pop up error when dir exist to avoid over written
    # os.mkdir(os.path.join(args.output, args.network_name))

    with tf.Session() as sess:
        dqn_agent = DQNAgent((q_network_online, q_network_target), preprocessor, memory, policy, args.n_actions,
                             args.gamma, args.target_update_freq, args.num_burn_in, args.train_freq, args.batch_size,
                             args.experience_replay, args.repetition_times, args.network_name, args.max_grad, args.env,
                             sess)

        optimizer = tf.train.AdamOptimizer(learning_rate=args.alpha)
        dqn_agent.compile(optimizer, mean_huber_loss)
        dqn_agent.fit(env, train_data, args.num_iterations, os.path.join(args.output, args.network_name),
                      args.save_freq, args.max_episode_length)


if __name__ == '__main__':
    main()
