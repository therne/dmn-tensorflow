import tensorflow as tf

from utils.nn import weight, bias


class EpisodeModule:
    """ Inner GRU module in episodic memory that creates episode vector. """
    def __init__(self, num_hidden, question, facts):
        self.question = question
        self.facts = facts

        # transposing for attention
        self.question_transposed = tf.transpose(question)
        self.facts_transposed = [tf.transpose(c) for c in facts]

        # parameters
        self.w1 = weight('w1', [num_hidden, 7 * num_hidden])
        self.b1 = bias('b1', [num_hidden, 1])
        self.w2 = weight('w2', [1, num_hidden])
        self.b2 = bias('b2', [1, 1])
        self.gru = tf.nn.rnn_cell.GRUCell(num_hidden)

    @property
    def init_state(self):
        return tf.zeros_like(self.facts[0])

    def new(self, memory):
        """ Creates new episode vector (will feed into Episodic Memory GRU)
        :param memory: Previous memory vector
        :return: episode vector
        """
        state = self.init_state
        memory = tf.transpose(memory)  # [N, D]
        for c, c_t in zip(self.facts, self.facts_transposed):
            g = self.attention(c_t, memory)
            state = g * self.gru(c, state)[0] + (1 - g) * state
            tf.get_variable_scope().reuse_variables()  # share params

        return state

    def attention(self, c, m):
        """ Attention mechanism. For details, see paper.
        :param c: A fact vector [N, D] at timestep
        :param m: Previous memory vector [N, D]
        :return: attention vector at timestep
        """
        with tf.variable_scope('attention'):
            # NOTE THAT instead of L1 norm we used L2
            q = self.question_transposed
            vec = tf.concat(0, [c, m, q, c*q, c*m, (c-q)**2, (c-m)**2])  # (7*d, N)

            # attention learning
            l1 = tf.matmul(self.w1, vec) + self.b1  # (N, d)
            l1 = tf.nn.tanh(l1)
            l2 = tf.matmul(self.w2, l1) + self.b2
            l2 = tf.nn.sigmoid(l2)
            return tf.transpose(l2)

        return att

