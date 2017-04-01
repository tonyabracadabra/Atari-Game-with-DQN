import tensorflow as tf
import numpy as np
import os
import gym
from gym import wrappers
import matplotlib.pyplot as plt
from pylab import *
from utils import *

"""Main DQN agent."""


class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """

    def __init__(self,
                 q_networks,
                 preprocessor,
                 memory,
                 policy,
                 num_actions,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 experience_replay,
                 repetition_times,
                 network_name,
                 max_grad,
                 env_name,
                 sess):

        self.q_network_online, self.q_network_target = q_networks
        self.target_vars = self.q_network_target.weights

        self.q_values_online = self.q_network_online.output
        self.q_values_target = self.q_network_target.output

        # Input placeholders for both online and target network
        self.state_online = self.q_network_online.input
        self.state_target = self.q_network_target.input

        self.preprocessor = preprocessor
        self.memory = memory
        self.gamma = gamma
        self.policy = policy
        self.num_actions = num_actions
        self.train_freq = train_freq
        self.num_burn_in = num_burn_in
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.experience_replay = experience_replay
        self.repetition_times = repetition_times
        self.network_name = network_name
        self.max_grad = max_grad
        self.env_name = env_name
        self.sess = sess

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """

        with tf.variable_scope('optimizer'):
            # print self.q_values_online.shape
            # Placeholder that we want to feed the updat in, just one value
            self.y_true = tf.placeholder(tf.float32, [None, ])
            # Placeholder that specify which action
            self.action = tf.placeholder(tf.int32, [None, ])
            # Transform it to one hot representation
            self.action_one_hot = tf.cast(tf.one_hot(self.action, depth=self.num_actions, \
                                                     on_value=1, off_value=0), tf.float32)

            # the output of the q_network is y_pred
            self.y_pred = tf.reduce_sum(tf.multiply(self.q_values_online, self.action_one_hot), axis=1)

            self.loss = loss_func(self.y_true, self.y_pred, self.max_grad)

            self.optimizer = optimizer.minimize(self.loss)

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        q_values_val = self.sess.run(self.q_values_online, feed_dict={self.state_online: state})

        return q_values_val

    def select_action(self, state, is_training):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        state = np.expand_dims(state, axis=0)
        q_values_val = self.calc_q_values(state)

        return self.policy.select_action(q_values_val, is_training)

    def _calc_y(self, next_states, rewards, not_terminal):
        y_vals = rewards
        # Calculating y values for deep q_network double
        if self.network_name is "deep_q_network_double" or self.network_name is "linear_q_network_double":
            actions = np.argmax(self.sess.run(self.q_values_online, \
                                              feed_dict={self.state_online: next_states}), axis=1)

            q_vals = self.gamma * self.sess.run(self.q_values_target, \
                                                feed_dict={self.state_target: next_states})

            added_vals = q_vals[np.arange(self.batch_size), actions]
        elif not self.experience_replay:
            # Calculating y values for no experience linear model
            added_vals = self.gamma * np.max(self.sess.run(self.q_values_online, \
                                                           feed_dict={self.state_online: next_states}), axis=1)
        else:
            # Calculating y values for other models
            added_vals = self.gamma * np.max(self.sess.run(self.q_values_target, \
                                                           feed_dict={self.state_target: next_states}), axis=1)

        y_vals[not_terminal] += added_vals[not_terminal]

        return y_vals

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """

        if self.experience_replay:
            states, next_states, actions, rewards, not_terminal = self.memory.sample(self.batch_size, self.preprocessor)
        else:
            states = np.stack(self.update_pool['states'])
            next_states = np.stack(self.update_pool['next_states'])
            actions = np.stack(self.update_pool['actions'])
            rewards = np.stack(self.update_pool['rewards'])
            not_terminal = self.update_pool['not_terminal']
            self.update_pool = {'actions': [], 'rewards': [], 'states': [], 'next_states': [], 'not_terminal': []}

        y_vals = self._calc_y(next_states, rewards, not_terminal)

        _, loss_val = self.sess.run([self.optimizer, self.loss], \
                                    feed_dict={self.state_online: states, self.y_true: y_vals, self.action: actions})

        return loss_val

    def _append_to_memory(self, curr_state, action, next_frame, reward, is_terminal):

        next_state = np.expand_dims(self.preprocessor.process_state_for_network(next_frame), axis=2)
        # plt.imshow(self.preprocessor.process_state_for_network(next_frame))
        # print self.preprocessor.process_state_for_network(next_frame)
        # plt.show()
        # append the next state to the last 3 frames in currstate to form the new state
        next_state = np.append(curr_state[:, :, 1:], next_state, axis=2)

        # Set s_{t+1} = s_t, a_t, x_{t+1} and preprocess phi_{t+1} = phi(s_{t+1})
        next_frame = self.preprocessor.process_state_for_memory(next_frame)

        if self.experience_replay:
            self.memory.append(next_frame, action, self.preprocessor.process_reward(reward), is_terminal)
        else:
            self.update_pool['states'].append(curr_state)
            self.update_pool['next_states'].append(next_state)
            self.update_pool['rewards'].append(reward)
            self.update_pool['actions'].append(action)
            self.update_pool['not_terminal'].append(not is_terminal)

        return next_state

    def fit(self, env, num_iterations, output_folder, save_freq=10000, max_episode_length=100):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """

        init = tf.global_variables_initializer()
        self.init_state = get_init_state(env, self.preprocessor)
        self.sess.run(init)
        env.reset()

        if not self.experience_replay:
            self.update_pool = {'actions': [], 'rewards': [], 'states': [], 'next_states': [], 'not_terminal': []}

        iter_t = 0
        episode_count = 0

        curr_state = self.init_state

        # Get the initial lives
        prev_lives = env.env.ale.lives()
        if self.experience_replay:
            print "Start filling up the replay memory before update ..."

            for j in xrange(self.num_burn_in):

                # action = self.select_action(curr_state, is_training=True)
                action = env.action_space.sample()

                # Execute action a_t in emulator and observe reward r_t and image x_{t+1}
                next_frame, reward, is_terminal, _ = env.step(action)

                life_terminal = False

                # Get current lives
                curr_lives = env.env.ale.lives()

                # check lives of agent, modify the original reward
                # if die, give -5 reward, if earn life, earn 50 reward
                if curr_lives < prev_lives:
                    life_terminal = True
                    reward = -5.0
                elif curr_lives > prev_lives:
                    reward = 50.0

                prev_lives = curr_lives

                # get next state while appending current frame to the memory
                next_state = self._append_to_memory(curr_state, action, next_frame, reward,
                                                    life_terminal or is_terminal)
                
                # If terminal, reset and goes back to the initial state
                if is_terminal:
                    env.reset()
                    curr_state = self.init_state
                else:
                    curr_state = next_state

            print "Has Prefilled the replay memory"

        while iter_t < num_iterations:
            # Get the initial state
            env.reset()
            curr_state = self.init_state
            action, total_reward, action_count = 0, 0, 0
            episode_count += 1
            prev_lives = env.env.ale.lives()

            print "Start " + str(episode_count) + "th Episode ..."
            for j in xrange(max_episode_length):
                # save model
                if iter_t % save_freq == 0:
                    self.evaluate_no_render()
                    model_json = self.q_network_online.to_json()
                    file_name = os.path.join(output_folder, str(iter_t) + ".json")
                    with open(file_name, "w") as json_file:
                        json_file.write(model_json)
                        # serialize weights to HDF5
                        self.q_network_online.save_weights(os.path.join(output_folder, str(iter_t) + ".h5"))
                    print("Saved model to disk")

                iter_t += 1
                if action_count == self.repetition_times:
                    action_count = 0
                    action = self.select_action(curr_state, is_training=True)
                action_count += 1

                # Execute action a_t in emulator and observe reward r_t and image x_{t+1}
                next_frame, reward, is_terminal, _ = env.step(action)
                life_terminal = False
                curr_lives = env.env.ale.lives()

                # update lives and reward correspondingly
                if curr_lives < prev_lives:
                    life_terminal = True
                    reward = -5.0
                elif curr_lives > prev_lives:
                    reward = 50.0

                prev_lives = curr_lives
                next_state = self._append_to_memory(curr_state, action, next_frame, reward,
                                                    life_terminal or is_terminal)
                total_reward += reward

                if is_terminal:
                    break

                # Time for updating (copy...) the target network
                if iter_t % self.target_update_freq == 0 and self.experience_replay:
                    get_hard_target_model_updates(self.q_network_target, self.q_network_online)

                if iter_t % self.train_freq == 0:
                    loss_val = self.update_policy()
                    if iter_t % 5000 == 0:
                        print str(iter_t) + "th iteration \n Loss val : " + str(loss_val)

                curr_state = next_state

            # update again after the episode ends...
            loss_val = self.update_policy()
            print str(episode_count) + "th Episode:\n" + "Reward: " + str(total_reward) + "\n Loss:" + str(loss_val)

    #
    def evaluate_no_render(self):
        """Test your agent with a provided environment.

        Evaluate the model 20 times every 100,000 interactions in training
        """
        num_episodes = 0
        env = gym.make(self.env_name)

        reward_avg = 0

        print "Start evaluating ... "
        while num_episodes < 20:
            env.reset()
            # Get the initial state
            curr_state = self.init_state

            is_terminal = False
            total_reward = 0
            while not is_terminal:
                action = self.select_action(curr_state, is_training=False)
                next_frame, reward, is_terminal, _ = env.step(action)

                # process and generate next state
                next_frame = self.preprocessor.process_state_for_network(next_frame)
                next_state = np.expand_dims(next_frame, axis=2)
                next_state = np.append(curr_state[:, :, 1:], next_state, axis=2)

                total_reward += reward

                curr_state = next_state
            
            reward_avg += total_reward
            num_episodes += 1

        reward_avg /= 20

        print "Average reward: " + str(reward_avg)

    def evaluate(self, env, log_file, num_episodes, render=False):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """

        # initialize
        self.init_state = get_init_state(env, self.preprocessor)
        env = wrappers.Monitor(env, log_file)
        rewards = []

        while num_episodes > 0:
            env.reset()
            # Get the initial state
            curr_state = self.init_state
            action = self.select_action(curr_state, is_training=False)
            is_terminal = False
            i = 0
            curr_reward = 0

            while not is_terminal:
                if render:
                    env.render()
                if i % self.repetition_times == 0:
                    action = self.select_action(curr_state, is_training=False)

                # take a step
                next_frame, reward, is_terminal, _ = env.step(action)
                curr_reward += reward

                # process and generate next state
                next_frame = self.preprocessor.process_state_for_network(next_frame)
                next_state = np.expand_dims(next_frame, axis=2)
                next_state = np.append(curr_state[:, :, 1:], next_state, axis=2)

                curr_state = next_state

                i += 1
            rewards.append(curr_reward)
            print "{}th Episode Finished".format(num_episodes)
            num_episodes -= 1

        print "\nStatistics:"
        print "Mean: {}".format(np.mean(rewards))
        print "Standard deviation: {}".format(np.std(rewards))

