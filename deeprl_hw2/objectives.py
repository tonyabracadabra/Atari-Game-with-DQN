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

    func = lambda i: tf.cond(tf.abs(y_true[i] - y_pred[i]) < max_grad, lambda: 0.5 * tf.square(y_true[i] - y_pred[i]), \
                             lambda: max_grad * (tf.abs(y_true[i] - y_pred[i]) - 0.5 * max_grad))

    if semver.match(tf.__version__, '<1.0.0'):
        result = tf.pack(map(func, xrange(y_true.shape[0].value)))
    else:
        result = tf.stack(map(func, xrange(y_true.shape[0].value)))

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
