import tensorflow as tf
from tensorflow.python.ops.nn import tanh
from utils.nn import weight, bias, batch_norm


class AttnGRU:
    """Attention-based Gated Recurrent Unit cell (cf. https://arxiv.org/abs/1603.01417)."""

    def __init__(self, num_units, is_training, bn):
        self._num_units = num_units
        self.is_training = is_training
        self.batch_norm = bn

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

        if self.batch_norm:
            with tf.variable_scope('Linear1'):
                x_w = batch_norm(tf.matmul(x, w), is_training=self.is_training)
            with tf.variable_scope('Linear2'):
                h_u = batch_norm(tf.matmul(h, u), is_training=self.is_training)
            return x_w + h_u + b
        else:
            return tf.matmul(x, w) + tf.matmul(h, u) + b
