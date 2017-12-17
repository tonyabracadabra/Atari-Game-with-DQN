import numpy as np
from deeprl.transition_dp.core.core import Preprocessor
from deeprl.transition_dp.core.dp_envs import State


class DependencyParsingPreprocessor(Preprocessor):
    """
    """

    def __init__(self, tok_manager):
        self.tok_manager = tok_manager
        self.padding = lambda a, pad, n: a + [pad] * (n - len(a))

    def process(self, state: State, ex: dict):
        # print('Before:', state)
        # print('ex', ex)

        stack, buf, arcs = state.stack, state.buffer, state.dependencies

        def get_lc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k],
                          reverse=True)

        # word feature
        features = [self.tok_manager.NULL] * (3 - len(stack)) + [ex['word'][x] for x in stack[-3:]]
        features += [ex['word'][x] for x in buf[:3]] + [self.tok_manager.NULL] * (3 - len(buf))
        # pos feature
        p_features = [self.tok_manager.P_NULL] * (3 - len(stack)) + [ex['pos'][x] for x in stack[-3:]]
        p_features += [ex['pos'][x] for x in buf[:3]] + [self.tok_manager.P_NULL] * (3 - len(buf))
        # left & right most node
        l_features = []
        for i in range(2):
            if i < len(stack):
                k = stack[-i - 1]
                lc = get_lc(k)
                rc = get_rc(k)
                llc = get_lc(lc[0]) if len(lc) > 0 else []
                rrc = get_rc(rc[0]) if len(rc) > 0 else []

                features.append(ex['word'][lc[0]] if len(lc) > 0 else self.tok_manager.NULL)
                features.append(ex['word'][rc[0]] if len(rc) > 0 else self.tok_manager.NULL)
                features.append(ex['word'][lc[1]] if len(lc) > 1 else self.tok_manager.NULL)
                features.append(ex['word'][rc[1]] if len(rc) > 1 else self.tok_manager.NULL)
                features.append(ex['word'][llc[0]] if len(llc) > 0 else self.tok_manager.NULL)
                features.append(ex['word'][rrc[0]] if len(rrc) > 0 else self.tok_manager.NULL)

                p_features.append(ex['pos'][lc[0]] if len(lc) > 0 else self.tok_manager.P_NULL)
                p_features.append(ex['pos'][rc[0]] if len(rc) > 0 else self.tok_manager.P_NULL)
                p_features.append(ex['pos'][lc[1]] if len(lc) > 1 else self.tok_manager.P_NULL)
                p_features.append(ex['pos'][rc[1]] if len(rc) > 1 else self.tok_manager.P_NULL)
                p_features.append(ex['pos'][llc[0]] if len(llc) > 0 else self.tok_manager.NULL)
                p_features.append(ex['pos'][rrc[0]] if len(rrc) > 0 else self.tok_manager.P_NULL)

                l_features.append(ex['label'][lc[0]] if len(lc) > 0 else self.tok_manager.L_NULL)
                l_features.append(ex['label'][rc[0]] if len(rc) > 0 else self.tok_manager.L_NULL)
                l_features.append(ex['label'][lc[1]] if len(lc) > 1 else self.tok_manager.L_NULL)
                l_features.append(ex['label'][rc[1]] if len(rc) > 1 else self.tok_manager.L_NULL)
                l_features.append(ex['label'][llc[0]] if len(llc) > 0 else self.tok_manager.L_NULL)
                l_features.append(ex['label'][rrc[0]] if len(rrc) > 0 else self.tok_manager.NULL)
            else:
                features += [self.tok_manager.NULL] * 6
                p_features += [self.tok_manager.P_NULL] * 6
                l_features += [self.tok_manager.L_NULL] * 6

        # print('after', np.array(features + p_features + l_features))

        return np.array(features + p_features + l_features)

    @staticmethod
    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return reward
        # return 1.0 if reward > 0 else -1.0
