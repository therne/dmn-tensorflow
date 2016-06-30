import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

from utils.nn import weight, bias
from utils.attn_gru import AttnGRU


class EpisodeModule(RNNCell):
    """ Inner GRU module in episodic memory that creates episode vector. """
    def __init__(self, num_hidden, question, is_training, bn):
        self.num_hidden = num_hidden
        self.question = question

        # parameters
        self.w1 = weight('w1', [4 * num_hidden, num_hidden])
        self.b1 = bias('b1', [num_hidden])
        self.w2 = weight('w2', [num_hidden, 1])
        self.b2 = bias('b2', [1])
        self.gru = AttnGRU(num_hidden, is_training, bn)

    @property
    def state_size(self):
        return self.num_hidden

    @property
    def output_size(self):
        return self.num_hidden

    def __call__(self, inputs, state, scope=None):
        """ Creates new contextual vector at each step """
        with tf.variable_scope("AttnGate"):
            g = self.attention(inputs, self.memory)
            state = self.gru(inputs, state, g)

        return state, state

    def attention(self, f, m):
        """ Attention mechanism. For details, see paper.
        :param f: A fact vector [N, D] at timestep
        :param m: Previous memory vector [N, D]
        :return: attention vector at timestep
        """
        with tf.variable_scope('attention'):
            # NOTE THAT instead of L1 norm we used L2
            q = self.question
            vec = tf.concat(1, [f * q, f * m, tf.abs(f - q), tf.abs(f - m)])  # [N, 4*d]

            # attention learning
            l1 = tf.matmul(vec, self.w1) + self.b1  # [N, d]
            l1 = tf.nn.tanh(l1)
            l2 = tf.matmul(l1, self.w2) + self.b2
            l2 = tf.nn.softmax(l2)
            return l2

        return att
