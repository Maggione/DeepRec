import math
import tensorflow as tf
import numpy as np
from src.base_model import BaseModel

__all__ = ["MKRModel"]


class MKRModel(BaseModel):
    """define Factorization-Machine based Neural Network Model"""

    def _build_graph(self, hparams):
        self.n_user = hparams.n_users
        self.n_item = hparams.n_items
        self.n_entity = hparams.n_entity
        self.n_relation = hparams.n_relation
        self.hparams = hparams
        self.regularizer = tf.contrib.layers.l2_regularizer(hparams.layer_l2)
        with tf.variable_scope("MKR") as scope:
            self._build_model()
            self._build_training()
        return self.logit

    @staticmethod
    def embedding_layer(name, shape, initializer):
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    @staticmethod
    def weight(name, dim):
        return tf.Variable(tf.truncated_normal([dim, 1]), name=name)

    @staticmethod
    def bias(name, dim):
        return tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[dim, 1]), name=name)

    @staticmethod
    def dense_layer(input_unit, dim, activation, regularizer):
        def _activate(activation):
            if activation == 'sigmoid':
                return tf.nn.sigmoid
            elif activation == 'softmax':
                return tf.nn.softmax
            elif activation == 'relu':
                return tf.nn.relu
            elif activation == 'tanh':
                return tf.nn.tanh
            elif activation == 'elu':
                return tf.nn.elu
            elif activation == 'identity':
                return tf.identity
            else:
                raise ValueError("this activations not defined {0}".format(activation))
        return tf.layers.dense(input_unit, dim, activation=_activate(activation),
                               kernel_regularizer=regularizer)

    def _build_model(self):
        with tf.name_scope('embedding'):
            self.user_embedding = self.embedding_layer('user', [self.n_user, self.hparams.dim], self.initializer)
            self.item_embedding = self.embedding_layer('item', [self.n_item, self.hparams.dim], self.initializer)
            self.entity_embedding = self.embedding_layer('entity', [self.n_entity, self.hparams.dim], self.initializer)
            self.relation_embedding = self.embedding_layer('relation', [self.n_relation, self.hparams.dim], self.initializer)

        # layer 0
        with tf.name_scope('rs'):
            self.user_0 = tf.nn.embedding_lookup(self.user_embedding, self.iterator.users)
            self.item_0 = tf.nn.embedding_lookup(self.item_embedding, self.iterator.items)
            self.entity_0 = tf.nn.embedding_lookup(self.entity_embedding, self.iterator.items)
            self.item_0 = tf.expand_dims(self.item_0, -1)
            self.entity_0 = tf.expand_dims(self.entity_0, -1)
            self.Crs_0 = tf.matmul(self.item_0, self.entity_0, transpose_b=True)

        with tf.name_scope('kg'):
            self.head_0 = tf.nn.embedding_lookup(self.entity_embedding, self.iterator.heads)
            self.relation_0 = tf.nn.embedding_lookup(self.relation_embedding, self.iterator.relations)
            self.tail = tf.nn.embedding_lookup(self.entity_embedding, self.iterator.tails)
            self.itemkg_0 = tf.nn.embedding_lookup(self.item_embedding, self.iterator.heads)
            self.itemkg_0 = tf.expand_dims(self.itemkg_0, -1)
            self.head_0 = tf.expand_dims(self.head_0, -1)
            self.Ckg_0 = tf.matmul(self.itemkg_0, self.head_0, transpose_b=True)

        """
        with tf.name_scope('cc'):
            self.item_0 = tf.expand_dims(self.item_0, -1)
            self.head_0 = tf.expand_dims(self.head_0, -1)
            self.C_0 = tf.matmul(self.item_0, self.head_0, transpose_b=True)
        """
        # layer 1
        with tf.name_scope('rs'):
            self.user_1 = self.dense_layer(self.user_0, self.hparams.dim, self.hparams.activation[0], self.regularizer)
            self.item_1 = \
                tf.reshape(
                    tf.matmul(tf.reshape(self.Crs_0, [-1, self.hparams.dim]), self.weight('w_0_VV', self.hparams.dim)),
                    [-1, self.hparams.dim, 1]) + \
                tf.reshape(
                    tf.matmul(tf.reshape(tf.transpose(self.Crs_0, perm=[0, 2, 1]), [-1, self.hparams.dim]),
                              self.weight('w_0_EV', self.hparams.dim)),
                    [-1, self.hparams.dim, 1]) + \
                self.bias('b_0_V', self.hparams.dim)
        with tf.name_scope('kg'):
            self.relation_1 = self.dense_layer(self.relation_0, self.hparams.dim, self.hparams.activation[0], self.regularizer)
            self.head_1 = \
                tf.reshape(
                    tf.matmul(tf.reshape(self.Ckg_0, [-1, self.hparams.dim]), self.weight('w_0_VE', self.hparams.dim)),
                    [-1, self.hparams.dim, 1]) + \
                tf.reshape(
                    tf.matmul(tf.reshape(tf.transpose(self.Ckg_0, perm=[0, 2, 1]), [-1, self.hparams.dim]),
                              self.weight('w_0_EE', self.hparams.dim)),
                    [-1, self.hparams.dim, 1]) + \
                self.bias('b_0_E', self.hparams.dim)
        # with tf.name_scope('cc'):
            # self.C_1 = tf.matmul(self.item_1, self.head_1, transpose_b=True)

        # final layer
        with tf.name_scope('rs'):
            self.logit = tf.reduce_sum(self.user_1 * tf.squeeze(self.item_1, -1), axis=1)

        with tf.name_scope('kg'):
            self.concat = tf.concat([tf.squeeze(self.head_1, -1), self.relation_1], axis=1)
            self.pred_tail = self.dense_layer(self.concat, self.hparams.dim, self.hparams.activation[0], self.regularizer)

    def _build_training(self):
        with tf.name_scope('training_for_rs'):
            self.base_loss_rs = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.iterator.labels, logits=self.logit))
            self.regularization_rs = (self.hparams.embed_l2 * tf.nn.l2_loss(self.user_0) +
                                      self.hparams.embed_l2 * tf.nn.l2_loss(self.item_0) +
                                      tf.losses.get_regularization_loss('rs')) / self.hparams.batch_size
            self.loss_rs = self.base_loss_rs + self.regularization_rs

        with tf.name_scope('training_for_kg'):
            self.score_kg = -tf.nn.sigmoid(tf.reduce_sum(self.tail * self.pred_tail, axis=1))
            self.regularization_kg = (self.hparams.embed_l2 * tf.nn.l2_loss(self.head_0) +
                                      self.hparams.embed_l2 * tf.nn.l2_loss(self.relation_0)+
                                      self.hparams.embed_l2 * tf.nn.l2_loss(self.tail) +
                                      tf.losses.get_regularization_loss('kg')) / self.hparams.batch_size
            self.loss_kg = self.score_kg + self.regularization_kg

        train_kg_opt = self._train_opt(self.hparams, self.hparams.lr_kg)
        train_rs_opt = self._train_opt(self.hparams, self.hparams.lr_rs)
        self.train_kg_step = train_kg_opt.minimize(self.loss_kg)
        self.train_rs_step = train_rs_opt.minimize(self.loss_rs)


    """
    def _build_evaluation(self):
        with tf.name_scope('evaluation'):
            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.iterator.labels, tf.round(self.pred)), tf.float32))
            _, self.auc = tf.metrics.auc(self.iterator.labels, self.pred)
    """


    def train(self, sess):
        if self.hparams.current_epoch % self.hparams.kg_training_interval == 1:
            step_result = sess.run([self.train_kg_step, self.loss_rs, self.base_loss_rs, self.merged])
        else:
            step_result = sess.run([self.train_rs_step, self.loss_rs, self.base_loss_rs, self.merged])

        return step_result

    def eval(self, sess):
        return sess.run([self.loss, self.data_loss, self.pred, self.iterator.labels])

    def infer(self, sess):
        return sess.run([self.pred])

    def _get_loss(self, hparams):
        self.data_loss = self.base_loss_rs
        self.regular_loss = self.regularization_rs
        self.loss = self.loss_rs
        return self.loss