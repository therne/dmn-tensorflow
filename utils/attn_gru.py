import tensorflow as tf
from tensorflow.python.ops.nn import tanh
from utils.nn import weight, bias


class AttnGRU:
    """Attention-based Gated Recurrent Unit cell (cf. https://arxiv.org/abs/1603.01417)."""

    def __init__(self, num_units):
        self._num_units = num_units

    def __call__(self, inputs, state, attention, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or 'AttrGRU'):
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset.
                r = tf.nn.sigmoid(self._linear(inputs, state, bias_default=1.0))
            with tf.variable_scope("Candidate"):
                c = tanh(self._linear(inputs, r * state))

            new_h = attention * c + (1 - attention) * state
        return new_h

    def _linear(self, x, h, bias_default=0.0):
        I, D = x.get_shape().as_list()[1], self._num_units
        w = weight('W', [I, D])
        u = weight('U', [D, D])
        b = bias('b', D, bias_default)
        return tf.matmul(x, w) + tf.matmul(h, u) + b
