import gym
from gym.spaces import prng
from gym.envs import register
from copy import deepcopy
import numpy as np
from abc import *


class State:
    """
    State class for Dependency parsing
    """

    def __init__(self, buffer: list, stack: list, dependencies: list):
        """

        :param buffer:
        :param stack:
        :param dependencies:
        """
        self.buffer = buffer
        self.stack = stack
        self.dependencies = dependencies

    def __str__(self):
        return 'State:\n' + list(zip(['Buffer', 'Stack', 'Dependencies'],
                                     [self.buffer, self.stack, self.dependencies])).__str__()[1:-1]


class DiscreteSpaceWithConstraint(gym.Space):
    """
    {0,1,...,n-1}
    Example usage:
    self.observation_space = spaces.Discrete(2)
    """

    def __init__(self, n, feasibility_func):
        self.n = n
        self.feasibility_func = feasibility_func

    def sample(self):
        return prng.np_random.randint(self.n)

    def sample_with_feasibility(self, state: State):
        feasibility = self.feasibility_func(state)
        indices = np.where(feasibility == 1)[0]

        a = np.random.choice(indices)
        # print('action', a)

        return a
        # return np.random.choice(indices)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return 0 <= as_int < self.n

    @property
    def shape(self):
        return (self.n,)

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n


class BaseTransitionEnv(gym.Env):
    """
    Base Transition Environment


    Methods
    -------
    init(ex: dict)
      Initialize the environment with a sentence sample
    __set_actions(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.

    """
    def __init__(self, dependency_labels_file):
        """Load the dependency labels file then set actions
        """
        with open(dependency_labels_file) as label_file:
            labels = [label.strip() for label in label_file]
            self.index2label = dict(zip(range(len(labels)), labels))
            self.label2index = dict(zip(labels, range(len(labels))))

        self.action2index = {}
        self.curr_state = self.init_state = State([], [], [])
        self.ex = {"word": [], "pos": [], "label": []}

    @abstractmethod
    def init(self, ex: dict):
        pass

    @abstractmethod
    def __set_actions(self):
        pass

    @abstractmethod
    def __cost(self, action):
        pass

    @abstractmethod
    def _step(self, action):
        pass

    @abstractmethod
    def get_action_feasibility(selfs, state: State):
        pass

    @property
    @abstractmethod
    def is_terminated(self):
        pass

    @property
    def n_actions(self):
        return len(self.action2index)

    @property
    def actions(self):
        return self.action2index.keys()

    @property
    def labels(self):
        return self.label2index.keys()

    @property
    def _n_labels(self):
        return len(self.label2index)

    def _reset(self):
        self.curr_state = deepcopy(self.init_state)


class ArcStandardTransitionEnv(BaseTransitionEnv):
    def __init__(self, dependency_labels_file):
        super().__init__(dependency_labels_file)
        self.__set_actions()
        self.action_space = DiscreteSpaceWithConstraint(self.n_actions, self.get_action_feasibility)

    def init(self, ex: dict):
        self.ex = ex
        self.curr_state = State(list(range(1, len(ex['word']))), [0], [])
        self.init_state = deepcopy(self.curr_state)

        return self.init_state

    def __set_actions(self):
        # noinspection PyTypeChecker
        actions = [('S',)] + sum([[('L', label), ('R', label)] for label in self.label2index.keys()], [])
        self.action2index = dict(zip(actions, range(len(actions))))
        self.index2action = dict(zip(range(len(actions)), actions))

    def _step(self, action):
        action = self.index2action[action]
        reward = -self.__cost(action)
        action_type, label = action[0], self.label2index[action[1]] if len(action) == 2 else None
        if action_type == 'S':
            pop_word = self.curr_state.buffer.pop(0)
            self.curr_state.stack.append(pop_word)
        elif action_type == 'L':
            self.curr_state.dependencies.append((self.curr_state.stack[-1], self.curr_state.stack[-2], label))
            del self.curr_state.stack[-2]
        elif action_type == 'R':
            self.curr_state.dependencies.append((self.curr_state.stack[-2], self.curr_state.stack[-1], label))
            del self.curr_state.stack[-1]

        return deepcopy(self.curr_state), reward, self.is_terminated

    def __cost(self, action):
        return 0

    def get_action_feasibility(self, state: State):
        pass

    @property
    def is_terminated(self):
        return len(self.curr_state.dependencies) == len(self.ex['word']) - 1


class ArcEagerTransitionEnv(BaseTransitionEnv):
    def __init__(self, dependency_labels_file):
        super().__init__(dependency_labels_file)
        self.__set_actions()
        self.action_space = DiscreteSpaceWithConstraint(self.n_actions, self.get_action_feasibility)

    def init(self, ex: dict):
        self.ex = ex
        self.curr_state = State(list(range(1, len(ex['word']))) + [0], [], [])
        self.init_state = deepcopy(self.curr_state)

        return self.curr_state

    def __set_actions(self):
        # noinspection PyTypeChecker
        actions = [('S',), ('P',)] + sum([[('L', label), ('R', label)] for label in self.label2index.keys()], [])
        self.action2index = dict(zip(actions, range(len(actions))))
        self.index2action = dict(zip(range(len(actions)), actions))

    def _step(self, action):
        # print(self.index2action)
        action = self.index2action[action]
        reward = float(-self.__cost(action))
        action_type, label = action[0], self.label2index[action[1]] if len(action) == 2 else None

        # print(action)
        # print(self.curr_state)

        is_valid_transition = True
        if action_type == 'S':
            # Unable to proceed
            # if len(self.curr_state.buffer) == 0:
            #     is_valid_transition = False
            # else:
            pop_word = self.curr_state.buffer.pop(0)
            self.curr_state.stack.append(pop_word)
        elif action_type == 'P':
            # Unable to proceed
            # if len(self.curr_state.stack) == 0:
            #     is_valid_transition = False
            # else:
            self.curr_state.stack.pop()
        elif action_type == 'L':
            # Unable to proceed
            # if len(self.curr_state.stack) == 0 or len(self.curr_state.buffer) == 0 or self.curr_state.stack[-1] == 0:
            #     is_valid_transition = False
            # else:
            # print(action)
            tail_word = self.curr_state.stack.pop()
            self.curr_state.dependencies.append((self.curr_state.buffer[0], tail_word, label))
        elif action_type == 'R':
            # Unable to proceed
            # if len(self.curr_state.buffer) == 0 or len(self.curr_state.stack) == 0:
            #     is_valid_transition = False
            # else:
            pop_word = self.curr_state.buffer.pop(0)
            head_word = self.curr_state.stack[-1]
            self.curr_state.stack.append(pop_word)
            self.curr_state.dependencies.append((head_word, pop_word, label))
        else:
            raise ValueError("Unknown Transition")

        if not is_valid_transition:
            reward = -10.0

        return deepcopy(self.curr_state), reward, self.is_terminated, is_valid_transition

    def __cost(self, action):
        ex = self.ex
        buffer = self.curr_state.buffer
        stack = self.curr_state.stack

        action_type, label = action[0], self.label2index[action[1]] if len(action) == 2 else None

        if action_type == 'S':
            if len(buffer) <= 1:
                return False
            buffer_top = buffer[0]
            cost_for_shift = 0
            for si in stack:
                if ex['head'][si] == buffer_top or ex['head'][buffer_top] == si:
                    cost_for_shift += 1
            return cost_for_shift

        if len(buffer) == 0 or len(stack) == 0:
            return 0
        buffer_top = buffer[0]
        stack_top = stack[-1]
        if action_type == 'P':  # reduce
            if buffer_top == 0:
                buffer_top = len(ex['word'])
            cost_for_pop = 0
            for bi in range(buffer_top, len(ex['word'])):
                if ex['head'][bi] == stack_top:
                    cost_for_pop += 1
            return cost_for_pop
        elif action_type == 'L':
            label_flag = label is None or ex['label'][buffer_top] == label
            cost_for_left = 0
            if ex['head'][stack_top] == buffer_top: return label_flag
            if buffer_top == 0: buffer_top = len(ex['word'])
            for bi in range(buffer_top, len(ex['word'])):
                if ex['head'][bi] == stack_top:
                    cost_for_left += 1
                if (not buffer_top == bi) and ex['head'][stack_top] == bi:
                    cost_for_left += 1
            if label is not None and not (ex['label'][stack_top] == label):
                cost_for_left += 0.5
            return cost_for_left
        elif action_type == 'R':
            label_flag = label is None or ex['label'][buffer_top] == label
            buffer_top_head = ex['head'][buffer_top]
            if buffer_top_head == stack_top:
                return 0  # cost is zero here
            k_b_cost = buffer_top_head in stack or buffer_top_head in buffer

            buffer_top_children_in_stack = [i for i in range(1, len(ex['word']))
                                            if ex['head'][i] == buffer_top and i in stack]

            if buffer_top_head not in buffer and buffer_top_head not in stack \
                    and len(buffer_top_children_in_stack) == 0:
                return 0  # here cost is 0
            if k_b_cost:
                return 1  # . here cost is 1

            dep_tail = [dep[1] for dep in self.curr_state.dependencies]

            cost_for_right = len([i for i in buffer_top_children_in_stack if i not in dep_tail])

            return cost_for_right

    def get_action_feasibility(self, state: State):
        # print('feasibility')
        feasibility = np.ones(len(self.action2index))
        if len(state.buffer) == 1:
            feasibility[self.action2index[('S',)]] = 0
            for action in self.action2index.keys():
                if action[0] == 'R':
                    feasibility[self.action2index[action]] = 0

        if len(state.stack) == 0:
            for action in self.action2index.keys():
                if action[0] == 'L' or action[0] == 'R':
                    # print(action, self.action2index[action])
                    feasibility[self.action2index[action]] = 0
            feasibility[self.action2index[('P',)]] = 0
        else:
            stack_head = state.stack[-1]
            dp = state.dependencies
            if len([arc for arc in dp if arc[1] == stack_head]) > 0:
                for action in self.action2index.keys():
                    if action[0] == 'L':
                        feasibility[self.action2index[action]] = 0
            else:
                feasibility[self.action2index[('P',)]] = 0
        return feasibility

    @property
    def is_terminated(self):
        return len(self.curr_state.buffer) == 0 or \
               len(self.curr_state.dependencies) == len(self.ex['word']) - 1


def register_envs(dependency_labels_file):
    register(
        id='ArcStandardTransitionEnv-v0',
        entry_point='deeprl.dp_envs:ArcStandardTransitionEnv',
        kwargs={'dependency_labels_file': dependency_labels_file})

    register(
        id='ArcEagerTransitionEnv-v0',
        entry_point='deeprl_hw1.queue_envs:QueueEnv',
        kwargs={'dependency_labels_file': dependency_labels_file})

