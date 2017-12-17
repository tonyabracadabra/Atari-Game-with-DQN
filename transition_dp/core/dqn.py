from pylab import *
from deeprl.transition_dp.core.utils import *
from deeprl.transition_dp.core.dp_envs import *

"""Main DQN agent."""


class DQNAgent:
    """Class implementing DQN.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: core.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: core.core.Memory
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
        """

        with tf.variable_scope('optimizer'):
            # print self.q_values_online.shape
            # Placeholder that we want to feed the updat in, just one value
            self.y_true = tf.placeholder(tf.float32, [None, ])
            # Placeholder that specify which action
            self.action = tf.placeholder(tf.int32, [None, ])
            # Transform it to one hot representation
            self.action_one_hot = tf.cast(tf.one_hot(self.action,
                                                     depth=self.num_actions, on_value=1, off_value=0), tf.float32)
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

    def select_action(self, state, env, is_training):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        Returns
        --------
        selected action
        """

        feasibility = env.get_action_feasibility(state)

        state = np.expand_dims(self.preprocessor.process(state, env.ex), axis=0)
        q_values_val = self.calc_q_values(state)

        # Mask out the invalid actions
        q_values_val = np.ma.array(q_values_val.tolist(), mask=-(feasibility-1))

        return self.policy.select_action(q_values_val, is_training)

    def _calc_y(self, next_states, rewards, is_terminals):
        y_vals = rewards
        not_terminals = np.invert(is_terminals)
        # Calculating y values for deep q_network double
        if self.network_name is "deep_q_network_double" or self.network_name is "linear_q_network_double":
            actions = np.argmax(self.sess.run(self.q_values_online,
                                              feed_dict={self.state_online: next_states}), axis=1)

            q_vals = self.gamma * self.sess.run(self.q_values_target,
                                                feed_dict={self.state_target: next_states})

            added_vals = q_vals[np.arange(self.batch_size), actions]
        elif not self.experience_replay:
            # Calculating y values for no experience linear model
            added_vals = self.gamma * np.max(self.sess.run(self.q_values_online,
                                                           feed_dict={self.state_online: next_states}), axis=1)
        else:
            # Calculating y values for other models
            added_vals = self.gamma * np.max(self.sess.run(self.q_values_target,
                                                           feed_dict={self.state_target: next_states}), axis=1)

        y_vals[not_terminals] += added_vals[not_terminals]

        return y_vals

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        """

        if self.experience_replay:
            states, actions, rewards, next_states, is_terminals = self.memory.sample(self.batch_size)
        else:
            states = np.stack(self.update_pool['states'])
            # print(states)
            next_states = np.stack(self.update_pool['next_states'])
            # print(next_states)

            actions = np.stack(self.update_pool['actions'])
            # print(actions)
            rewards = np.stack(self.update_pool['rewards'])
            # print(rewards)

            is_terminals = np.stack(self.update_pool['is_terminals'])
            # print(not_terminal)
            self.update_pool = {'actions': [], 'rewards': [], 'states': [], 'next_states': [], 'is_terminals': []}

        y_vals = self._calc_y(next_states, rewards, is_terminals)

        _, loss_val = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.state_online: states, self.y_true: y_vals, self.action: actions})

        return loss_val

    def _append_to_memory(self, curr_state, action, reward, next_state, is_terminal):
        # Set s_{t+1} = s_t, a_t, x_{t+1} and preprocess phi_{t+1} = phi(s_{t+1})
        if self.experience_replay:
            self.memory.append(curr_state, action, reward, next_state, is_terminal)
        else:
            self.update_pool['states'].append(curr_state)
            self.update_pool['next_states'].append(next_state)
            self.update_pool['rewards'].append(reward)
            self.update_pool['actions'].append(action)
            self.update_pool['not_terminal'].append(not is_terminal)

        return next_state

    def fit(self, env: BaseTransitionEnv, train_data: list, num_iterations, output_folder, save_freq=10000,
            max_episode_length=100):
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
        env: BaseTransitionEnv
            Dependency Parsing Environment
        train_data: list
            Training sentences where features are extracted from
        num_iterations: int
            How many samples/updates to perform.
        max_episode_length: int
            How long a single episode should last before the agent
            resets.
        save_freq: int
            Saving frequency
        max_episode_length: int
            Maximum number of actions for one sentence
        """

        init = tf.global_variables_initializer()
        data = data_generator(train_data)

        self.sess.run(init)
        env.reset()

        if not self.experience_replay:
            self.update_pool = {'actions': [], 'rewards': [], 'states': [], 'next_states': [], 'not_terminal': []}

        iter_t = 0
        episode_count = 0

        ''' Experience Replay '''
        # Get the initial state for dependency parsing
        curr_state_raw = env.init(next(data))
        curr_state = self.preprocessor.process(curr_state_raw, env.ex)
        if self.experience_replay:
            print("Start filling up the replay memory before update ...")

            for j in range(self.num_burn_in):
                # action = self.select_action(curr_state, is_training=True)
                action = env.action_space.sample_with_feasibility(curr_state_raw)
                # Execute action a_t in emulator and observe reward r_t and image x_{t+1}
                next_state_raw, reward, is_terminal, _ = env.step(action)
                next_state = self.preprocessor.process(next_state_raw, env.ex)

                # TODO
                # if reward/cost is too small, consider is_terminal = False???
                self._append_to_memory(curr_state, action, reward, next_state, is_terminal)

                # If terminal, reset and goes back to the initial state
                if is_terminal:
                    curr_state_raw = env.init(next(data))
                    curr_state = self.preprocessor.process(curr_state_raw, env.ex)
                else:
                    curr_state_raw, curr_state = next_state_raw, next_state

            print("Has Pre-filled the replay memory")

        while iter_t < num_iterations:
            # Get the initial state for dependency parsing
            curr_state_raw = env.init(next(data))
            curr_state = self.preprocessor.process(curr_state_raw, env.ex)

            action, total_reward, action_count = 0, 0, 0
            episode_count += 1
            print("Start " + str(episode_count) + "th Episode ...")
            for j in range(max_episode_length):

                # save model
                # TODO
                # if iter_t % save_freq == 0:
                #     # self.evaluate_no_render()
                #     model_json = self.q_network_online.to_json()
                #     file_name = os.path.join(output_folder, str(iter_t) + ".json")
                #     with open(file_name, "w") as json_file:
                #         json_file.write(model_json)
                #         # serialize weights to HDF5
                #         self.q_network_online.save_weights(os.path.join(output_folder, str(iter_t) + ".h5"))
                #     print("Saved model to disk")

                iter_t += 1

                if action_count == self.repetition_times:
                    action_count = 0
                action_count += 1

                action = self.select_action(curr_state_raw, env, is_training=True)
                # Execute action a_t in emulator and observe reward r_t and state x_{t+1}
                next_state_raw, reward, is_terminal, is_valid_transition = env.step(action)
                next_state = self.preprocessor.process(next_state_raw, env.ex)

                # TODO
                # if reward/cost is too small, consider is_terminal = False???
                self._append_to_memory(curr_state, action, reward, next_state, is_terminal)
                total_reward += reward

                if is_terminal:
                    break

                # Time for updating (copy...) the target network
                if iter_t % self.target_update_freq == 0 and self.experience_replay:
                    get_hard_target_model_updates(self.q_network_target, self.q_network_online)

                if iter_t != 1 and iter_t % self.train_freq == 0:
                    loss_val = self.update_policy()
                    if iter_t % 5000 == 0:
                        print(str(iter_t) + "th iteration \n Loss val : " + str(loss_val))

                curr_state, curr_state_raw = next_state, next_state_raw

            # update again after the episode ends...
            if not self.experience_replay and len(self.update_pool['states']) != 0:
                loss_val = self.update_policy()
            print(str(episode_count) + "th Episode:\n" + "Reward: " + str(total_reward) + "\n Loss:" + str(loss_val))

    def evaluate(self, env: BaseTransitionEnv, test_data):
        """Test your agent with a provided environment.
        """

        '''
        Get predicted dependencies for each sentence
        '''
        dependencies = []
        for ex in test_data:
            is_terminal = False
            curr_state = self.preprocessor.process(env.init(ex), env.ex)
            while is_terminal is False:
                action = self.select_action(curr_state, env, is_training=False)
                curr_state, _, is_terminal = env.step(action)
                curr_state = self.preprocessor.process(curr_state, env.ex)
            dependencies.append(env.curr_state.dependencies)

        '''
        Calculate the UAS and LAS scores
        '''

        UAS = LAS = all_tokens = 0.0
        for i, ex in enumerate(test_data):
            head = [-1] * len(ex['word'])
            label = [-1] * len(ex['word'])
            for h, t, l in dependencies[i]:
                head[t] = h
                label[t] = l
            for pred_h, gold_h, pred_l, gold_l in zip(head[1:], ex['head'][1:], label[1:], ex['label'][1:]):
                UAS += 1 if pred_h == gold_h else 0
                LAS += 1 if pred_h == gold_h and gold_l == pred_l else 0
                all_tokens += 1
        UAS /= all_tokens
        LAS /= all_tokens

        print("\nStatistics:")
        print("UAS: {}".format(UAS))
        print("LAS: {}".format(LAS))

        return UAS, LAS, dependencies
