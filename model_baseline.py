from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
try:
    import tfplot
except:
    pass

from ops import conv2d, fc
from util import log

from vqa_util import question2str, answer2str


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.img_size = self.config.data_info[0]
        self.c_dim = self.config.data_info[2]
        self.q_dim = self.config.data_info[3]
        self.a_dim = self.config.data_info[4]
        self.conv_info = self.config.conv_info

        # create placeholders for the input
        self.img = tf.placeholder(
            name='img', dtype=tf.float32,
            shape=[self.batch_size, self.img_size, self.img_size, self.c_dim],
        )
        self.q = tf.placeholder(
            name='q', dtype=tf.float32, shape=[self.batch_size, self.q_dim],
        )
        self.a = tf.placeholder(
            name='a', dtype=tf.float32, shape=[self.batch_size, self.a_dim],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.img: batch_chunk['img'],  # [B, h, w, c]
            self.q: batch_chunk['q'],  # [B, n]
            self.a: batch_chunk['a'],  # [B, m]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd

    def build(self, is_train=True):

        n = self.a_dim
        conv_info = self.conv_info

        # build loss and accuracy {{{
        def build_loss(logits, labels):
            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy
        # }}}

        # Classifier: takes images as input and outputs class label [B, m]
        def C(img, q, scope='Classifier'):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                conv_1 = conv2d(img, conv_info[0], is_train, s_h=3, s_w=3, name='conv_1')
                conv_2 = conv2d(conv_1, conv_info[1], is_train, s_h=3, s_w=3, name='conv_2')
                conv_3 = conv2d(conv_2, conv_info[2], is_train, name='conv_3')
                conv_4 = conv2d(conv_3, conv_info[3], is_train, name='conv_4')
                conv_q = tf.concat([tf.reshape(conv_4, [self.batch_size, -1]), q], axis=1)
                fc_1 = fc(conv_q, 256, name='fc_1')
                fc_2 = fc(fc_1, 256, name='fc_2')
                fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=is_train, scope='fc_3/')
                fc_3 = fc(fc_2, n, activation_fn=None, name='fc_3')
                return fc_3

        logits = C(self.img, self.q, scope='Classifier')
        self.all_preds = tf.nn.softmax(logits)
        self.loss, self.accuracy = build_loss(logits, self.a)

        # Add summaries
        def draw_iqa(img, q, target_a, pred_a):
            fig, ax = tfplot.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_title(question2str(q))
            ax.set_xlabel(answer2str(target_a)+answer2str(pred_a, 'Predicted'))
            return fig

        try:
            tfplot.summary.plot_many('IQA/',
                                     draw_iqa, [self.img, self.q, self.a, self.all_preds],
                                     max_outputs=3,
                                     collections=["plot_summaries"])
        except:
            pass

        tf.summary.scalar("loss/accuracy", self.accuracy)
        tf.summary.scalar("loss/cross_entropy", self.loss)
        log.warn('Successfully loaded the model.')
