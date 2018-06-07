"""define Factorization-Machine based Neural Network Model"""
import math
import tensorflow as tf
import numpy as np
from src.base_model import BaseModel

__all__ = ["FmModel"]


class FmModel(BaseModel):
    """define Factorization-Machine based Neural Network Model"""

    def _build_graph(self, hparams):
        # dumpy variable
        self.keep_prob_train = 1 - np.array([1.0])
        self.keep_prob_test = np.ones_like([1.0])
        self.layer_keeps = tf.placeholder(tf.float32)
        with tf.variable_scope("Fm") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(name='embedding_layer',
                                                 shape=[hparams.FEATURE_COUNT, hparams.dim],
                                                 dtype=tf.float32)
                self.embed_params.append(self.embedding)
            logit = self._build_linear(hparams)
            logit = tf.add(logit, self._build_fm(hparams))
            return logit

    def _build_linear(self, hparams):
        with tf.variable_scope("linear_part", initializer=self.initializer) as scope:
            w_linear = tf.get_variable(name='w',
                                       shape=[hparams.FEATURE_COUNT, 1],
                                       dtype=tf.float32)
            b_linear = tf.get_variable(name='b',
                                       shape=[1],
                                       dtype=tf.float32)
            x = tf.SparseTensor(self.iterator.fm_feat_indices,
                                self.iterator.fm_feat_values,
                                self.iterator.fm_feat_shape)
            linear_output = tf.add(tf.sparse_tensor_dense_matmul(x, w_linear), b_linear)
            self.layer_params.append(w_linear)
            self.layer_params.append(b_linear)
            tf.summary.histogram("linear_part/w", w_linear)
            tf.summary.histogram("linear_part/b", b_linear)
            return linear_output

    def _build_fm(self, hparams):
        with tf.variable_scope("fm_part") as scope:
            x = tf.SparseTensor(self.iterator.fm_feat_indices,
                                self.iterator.fm_feat_values,
                                self.iterator.fm_feat_shape)
            xx = tf.SparseTensor(self.iterator.fm_feat_indices,
                                 tf.pow(self.iterator.fm_feat_values, 2),
                                 self.iterator.fm_feat_shape)
            fm_output = 0.5 * tf.reduce_sum(
                tf.pow(tf.sparse_tensor_dense_matmul(x, self.embedding), 2) - \
                tf.sparse_tensor_dense_matmul(xx,
                                              tf.pow(self.embedding, 2)), 1,
                keep_dims=True)
            return fm_output
