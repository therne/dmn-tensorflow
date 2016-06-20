import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq, rnn_cell

from models.base_model import BaseModel
from models.old.episode_module import EpisodeModule
from utils.nn import weight, batch_norm, dropout


class DMN(BaseModel):
    """ Dynamic Memory Networks (http://arxiv.org/abs/1506.07285)
        Semantic Memory version: Instead of implementing embedding layer,
        it uses GloVe instead. (First version of DMN paper.)
    """
    def build(self):
        params = self.params
        N, L, Q, F = params.batch_size, params.max_sent_size, params.max_ques_size, params.max_fact_count
        V, d, A = params.glove_size, params.hidden_size, self.words.vocab_size

        # initialize self
        # placeholders
        input = tf.placeholder(tf.float32, shape=[N, L, V], name='x')  # [num_batch, sentence_len, glove_dim]
        question = tf.placeholder(tf.float32, shape=[N, Q, V], name='q')  # [num_batch, sentence_len, glove_dim]
        answer = tf.placeholder(tf.int64, shape=[N], name='y')  # [num_batch] - one word answer
        input_mask = tf.placeholder(tf.bool, shape=[N, L], name='x_mask')  # [num_batch, sentence_len]
        is_training = tf.placeholder(tf.bool)

        # Prepare parameters
        gru = rnn_cell.GRUCell(d)

        # Input module
        with tf.variable_scope('input') as scope:
            input_list = self.make_decoder_batch_input(input)
            input_states, _ = seq2seq.rnn_decoder(input_list, gru.zero_state(N, tf.float32), gru)

            # Question module
            scope.reuse_variables()

            ques_list = self.make_decoder_batch_input(question)
            questions, _ = seq2seq.rnn_decoder(ques_list, gru.zero_state(N, tf.float32), gru)
            question_vec = questions[-1]  # use final state

        # Masking: to extract fact vectors at end of sentence. (details in paper)
        input_states = tf.transpose(tf.pack(input_states), [1, 0, 2])  # [N, L, D]
        facts = []
        for n in range(N):
            filtered = tf.boolean_mask(input_states[n, :, :], input_mask[n, :])  # [?, D]
            padding = tf.zeros(tf.pack([F - tf.shape(filtered)[0], d]))
            facts.append(tf.concat(0, [filtered, padding]))  # [F, D]

        facked = tf.pack(facts)  # packing for transpose... I hate TF so much
        facts = tf.unpack(tf.transpose(facked, [1, 0, 2]), num=F)  # F x [N, D]

        # Episodic Memory
        with tf.variable_scope('episodic') as scope:
            episode = EpisodeModule(d, question_vec, facts)

            memory = tf.identity(question_vec)
            for t in range(params.memory_step):
                memory = gru(episode.new(memory), memory)[0]
                scope.reuse_variables()

        # Regularizations
        if params.batch_norm:
            memory = batch_norm(memory, is_training=is_training)
        memory = dropout(memory, params.keep_prob, is_training)

        with tf.name_scope('Answer'):
            # Answer module : feed-forward version (for it is one word answer)
            w_a = weight('w_a', [d, A])
            logits = tf.matmul(memory, w_a)  # [N, A]

        with tf.name_scope('Loss'):
            # Cross-Entropy loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, answer)
            loss = tf.reduce_mean(cross_entropy)
            total_loss = loss + params.weight_decay * tf.add_n(tf.get_collection('l2'))

        with tf.variable_scope('Accuracy'):
            # Accuracy
            predicts = tf.cast(tf.argmax(logits, 1), 'int32')
            corrects = tf.equal(predicts, answer)
            num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
            accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        # Training
        optimizer = tf.train.AdadeltaOptimizer(params.learning_rate)
        opt_op = optimizer.minimize(total_loss, global_step=self.global_step)

        # placeholders
        self.x = input
        self.q = question
        self.y = answer
        self.mask = input_mask
        self.is_training = is_training

        # tensors
        self.total_loss = total_loss
        self.num_corrects = num_corrects
        self.accuracy = accuracy
        self.opt_op = opt_op

    def make_decoder_batch_input(self, input):
        """ Reshape batch data to be compatible with Seq2Seq RNN decoder.
        :param input: Input 3D tensor that has shape [num_batch, sentence_len, wordvec_dim]
        :return: list of 2D tensor that has shape [num_batch, wordvec_dim]
        """
        input_transposed = tf.transpose(input, [1, 0, 2])  # [L, N, V]
        return tf.unpack(input_transposed)

    def preprocess_batch(self, batches):
        """ Vectorizes padding and masks last word of sentence. (EOS token)
        :param batches: A tuple (input, question, label, mask)
        :return A tuple (input, question, label, mask)
        """
        params = self.params
        input, question, label = batches
        N, Q, F, V = params.batch_size, params.max_ques_size, params.max_fact_count, params.embed_size

        # calculate max sentence size
        L = 0
        for n in range(N):
            sent_len = np.sum([len(sentence) for sentence in input[n]])
            L = max(L, sent_len)
        params.max_sent_size = L

        # make input and question fixed size
        new_input = np.zeros([N, L, V])  # zero padding
        new_question = np.zeros([N, Q, V])
        new_mask = np.full([N, L], False, dtype=bool)
        new_labels = []

        for n in range(N):
            sentence = np.array(input[n]).flatten()  # concat all sentences
            sentence_len = len(sentence)

            input_mask = [index for index, w in enumerate(sentence) if w == '.']
            new_input[n, :sentence_len] = [self.words.vectorize(w) for w in sentence]

            sentence_len = len(question[n])
            new_question[n, :sentence_len] = [self.words.vectorize(w) for w in question[n]]

            new_labels.append(self.words.word_to_index(label[n]))

            # mask on
            for eos_index in input_mask:
                new_mask[n, eos_index] = True

        return new_input, new_question, new_labels, new_mask

    def get_feed_dict(self, batches, is_train):
        input, question, label, mask = self.preprocess_batch(batches)
        return {
            self.x: input,
            self.q: question,
            self.y: label,
            self.mask: mask,
            self.is_training: is_train
        }
