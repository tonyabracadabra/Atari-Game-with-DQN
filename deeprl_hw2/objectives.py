"""Loss functions."""

import tensorflow as tf
import semver


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """

    delta = tf.abs(y_true - y_pred)
    default = tf.constant(0.0)

    less_cond = tf.boolean_mask(delta, delta < max_grad)
    greater_cond = tf.boolean_mask(delta, delta >= max_grad)

    less = 0.5 * tf.square(less_cond)
    greater = max_grad * greater_cond - 0.5 * max_grad * max_grad

    result = tf.concat([less, greater], axis=0)

    return result
    
def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    return tf.reduce_mean(huber_loss(y_true, y_pred, max_grad))
