import gym
from gym.spaces import prng
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
