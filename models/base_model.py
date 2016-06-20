import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class BaseModel(object):
    """ Code from mem2nn-tensorflow. """
    def __init__(self, params, words):
        self.params = params
        self.words = words
        self.save_dir = params.save_dir

        with tf.variable_scope('DMN'):
            print("Building DMN...")
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.build()
            self.saver = tf.train.Saver()

    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, is_train):
        raise NotImplementedError()

    def train_batch(self, sess, batch):
        feed_dict = self.get_feed_dict(batch, is_train=True)
        return sess.run([self.opt_op, self.global_step], feed_dict=feed_dict)

    def test_batch(self, sess, batch):
        feed_dict = self.get_feed_dict(batch, is_train=False)
        return sess.run([self.num_corrects, self.total_loss, self.global_step], feed_dict=feed_dict)

    def train(self, sess, train_data, val_data):
        params = self.params
        num_epochs = params.num_epochs
        num_batches = train_data.num_batches

        print("Training %d epochs ..." % num_epochs)
        for epoch_no in tqdm(range(num_epochs), desc='Epoch', maxinterval=86400, ncols=100):
            for _ in range(num_batches):
                batch = train_data.next_batch()
                _, global_step = self.train_batch(sess, batch)

            train_data.reset()

            if (epoch_no + 1) % params.acc_period == 0:
                print()  # Newline for TQDM
                self.eval(sess, train_data, name='Training')

            if val_data and (epoch_no + 1) % params.val_period == 0:
                self.eval(sess, val_data, name='Validation')

            if (epoch_no + 1) % params.save_period == 0:
                self.save(sess)

        print("Training completed.")

    def eval(self, sess, data, name):
        num_batches = data.num_batches
        num_corrects = total = 0
        losses = []
        for _ in range(num_batches):
            batch = data.next_batch()
            cur_num_corrects, cur_loss, global_step = self.test_batch(sess, batch)
            num_corrects += cur_num_corrects
            total += len(batch[0])
            losses.append(cur_loss)
        data.reset()
        loss = np.mean(losses)

        print("[%s] step %d: Accuracy = %.2f%% (%d / %d), Loss = %.4f" % \
              (name, global_step, 100 * float(num_corrects) / total, num_corrects, total, loss))
        return loss

    def save(self, sess):
        print("Saving model to %s" % self.save_dir)
        self.saver.save(sess, self.save_dir, self.global_step)

    def load(self, sess):
        print("Loading model ...")
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)
