"""Common functions you may find useful in your implementation."""

import semver
import tensorflow as tf
import numpy as np


def get_uninitialized_variables(variables=None):
    """Return a list of uninitialized tf variables.

    Parameters
    ----------
    variables: tf.Variable, list(tf.Variable), optional
      Filter variable list to only those that are uninitialized. If no
      variables are specified the list of all variables in the graph
      will be used.

    Returns
    -------
    list(tf.Variable)
      List of uninitialized tf variables.
    """
    sess = tf.get_default_session()
    if variables is None:
        variables = tf.global_variables()
    else:
        variables = list(variables)

    if len(variables) == 0:
        return []

    if semver.match(tf.__version__, '<1.0.0'):
        init_flag = sess.run(
            tf.pack([tf.is_variable_initialized(v) for v in variables]))
    else:
        init_flag = sess.run(
            tf.stack([tf.is_variable_initialized(v) for v in variables]))

    return [v for v, f in zip(variables, init_flag) if not f]


# Tears of the debugging...
def initialize_updates_operations(target_vars):
    # placeholders for updating the online network
    update_phs = [tf.placeholder(tf.float32, shape=var.get_shape()) for var in target_vars]
    # update operations
    update_ops = [update_pair[0].assign(update_pair[1]) \
                  for update_pair in zip(target_vars, update_phs)]

    return update_phs, update_ops


def get_soft_target_model_updates(target, source, tau):
    """Return list of target model update ops.

    These are soft target updates. Meaning that the target values are
    slowly adjusted, rather than directly copied over from the source
    model.

    The update is of the form:

    $W' \gets (1- \tau) W' + \tau W$ where $W'$ is the target weight
    and $W$ is the source weight.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.
    tau: float
      The weight of the source weights to the target weights used
      during update.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """

    weights = map(lambda (x, y): x + y, zip(map(lambda w: (1 - tau) * w, target.get_weights()), \
                                            map(lambda w: tau * w, source.get_weights())))

    target.set_weights(weights)


def get_hard_target_model_updates(target, source):
    """Return list of target model update ops.

    These are hard target updates. The source weights are copied
    directly to the target network.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    # updating the parameters from the previous network
    target.set_weights(source.get_weights())


def get_init_state(env, preprocessor):
    env.reset()

    init_state = np.stack(map(preprocessor.process_state_for_network, \
                              [env.step(0)[0] for _ in xrange(4)]), axis=2)

    return init_state
