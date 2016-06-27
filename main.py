#!/usr/bin/env python3
import os

import tensorflow as tf

from read_data import read_babi, get_max_sizes
from utils.data_utils import load_glove, WordTable

flags = tf.app.flags

# directories
flags.DEFINE_string('model', 'dmn+', 'Model type - dmn+, dmn, dmn_embed [Default: DMN+]')
flags.DEFINE_boolean('test', False, 'true for testing, false for training [False]')
flags.DEFINE_string('data_dir', 'data/tasks_1-20_v1-2/en-10k', 'Data directory [data/tasks_1-20_v1-2/en-10k]')
flags.DEFINE_string('save_dir', 'save', 'Save path [save]')

# training options
flags.DEFINE_bool('gpu', True, 'Use GPU? [True]')
flags.DEFINE_integer('batch_size', 128, 'Batch size during training and testing [128]')
flags.DEFINE_integer('num_epochs', 256, 'Number of epochs for training [256]')
flags.DEFINE_float('learning_rate', 0.002, 'Learning rate [0.002]')
flags.DEFINE_boolean('load', False, 'Start training from saved model? [False]')
flags.DEFINE_integer('acc_period', 10, 'Training accuracy display period [10]')
flags.DEFINE_integer('val_period', 40, 'Validation period (for display purpose) [40]')
flags.DEFINE_integer('save_period', 80, 'Save period [80]')

# model params
flags.DEFINE_integer('memory_step', 3, 'Episodic Memory steps [3]')
flags.DEFINE_string('memory_update', 'relu', 'Episodic meory update method - relu or gru [relu]')
# flags.DEFINE_bool('memory_tied', False, 'Share memory update weights among the layers? [False]')
flags.DEFINE_integer('glove_size', 50, 'GloVe size - Only used in dmn [50]')
flags.DEFINE_integer('embed_size', 80, 'Word embedding size - Used in dmn+, dmn_embed [80]')
flags.DEFINE_integer('hidden_size', 80, 'Size of hidden units [80]')

# train hyperparameters
flags.DEFINE_float('weight_decay', 0.001, 'Weight decay - 0 to turn off L2 regularization [0.001]')
flags.DEFINE_float('keep_prob', 1., 'Dropout rate - 1.0 to turn off [1.0]')
flags.DEFINE_bool('batch_norm', True, 'Use batch normalization? [True]')

# bAbi dataset params
flags.DEFINE_integer('task', 1, 'bAbi Task number [1]')
flags.DEFINE_float('val_ratio', 0.1, 'Validation data ratio to training data [0.1]')

FLAGS = flags.FLAGS


def main(_):
    if FLAGS.model == 'dmn':
        word2vec = load_glove(FLAGS.glove_size)
        words = WordTable(word2vec, FLAGS.glove_size)
        from models.old.dmn import DMN

    elif FLAGS.model == 'dmn+':
        words = WordTable()
        from models.new.dmn_plus import DMN

    elif FLAGS.model == 'dmn_embed':
        words = WordTable()
        from models.old.dmn_embedding import DMN
    else:
        print('Unknown model type: %s' % FLAGS.model)
        return

    # Read data
    train = read_babi(FLAGS.data_dir, FLAGS.task, 'train', FLAGS.batch_size, words)
    test = read_babi(FLAGS.data_dir, FLAGS.task, 'test', FLAGS.batch_size, words)
    val = train.split_dataset(FLAGS.val_ratio)

    FLAGS.max_sent_size, FLAGS.max_ques_size, FLAGS.max_fact_count = get_max_sizes(train, test, val)
    print('Word count: %d, Max sentence len : %d' % (words.vocab_size, FLAGS.max_sent_size))

    # Modify save dir
    FLAGS.save_dir += '/task_%d/' % FLAGS.task
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir, exist_ok=True)

    with tf.Session() as sess:
        model = DMN(FLAGS, words)
        sess.run(tf.initialize_all_variables())

        if FLAGS.test:
            model.load(sess)
            model.eval(sess, test, name='Test')
        else:
            if FLAGS.load: model.load(sess)
            model.train(sess, train, val)

if __name__ == '__main__':
    tf.app.run()
