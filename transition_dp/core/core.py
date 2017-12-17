"""Core classes."""

import random
import numpy as np
from abc import *


class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuples

    Parameters
    ----------
    curr_state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """

    def __init__(self, curr_state, action, reward, next_state, is_terminal):
        self.curr_state = curr_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal


class Preprocessor:
    """Preprocessor base class.
    """

    @abstractmethod
    def process(self, state, ex):
         pass

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass


class ReplayMemory:
    """Interface for replay memories.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
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

    def __init__(self, max_size):
        """Setup memory.

        """
        self.max_size = max_size
        self.index = 0
        self._samples = []

    def append(self, curr_state, action, reward, next_state, is_terminal):
        # Maximizing over the previous frame to remove flickering
        sample = Sample(curr_state, action, reward, next_state, is_terminal)

        if len(self._samples) == self.max_size:
            self._samples[self.index] = sample
        else:
            self._samples.append(sample)

        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        random_indexes = set()
        # Select batch size valid indexes
        while len(random_indexes) < batch_size:
            new_random_indexes = random.sample(range(len(self._samples)), batch_size - len(random_indexes))
            random_indexes = random_indexes.union(new_random_indexes)

        # construct current states, next states, actions, rewards by stacking
        # different fields of the samples together
        random_samples = [self._samples[i] for i in random_indexes]

        curr_states = np.stack([s.curr_state for s in random_samples])
        next_states = np.stack([s.next_state for s in random_samples])

        actions = np.stack([s.action for s in random_samples])
        rewards = np.stack([s.reward for s in random_samples])
        is_terminals = np.stack([s.is_terminal for s in random_samples])

        return curr_states, actions, rewards, next_states, is_terminals

    def clear(self):
        self._samples = []
        self.index = 0

    def __iter__(self):
        return iter(self._samples)

    def __getitem__(self, key):
        return self._samples[key]

    def __len__(self):
        return len(self._samples)
