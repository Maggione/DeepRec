"""define Factorization-Machine based Neural Network Model"""
import math
import numpy as np
import tensorflow as tf
from src.base_model import BaseModel

__all__ = ["CCCFModel"]


class CCCFModel(BaseModel):
    """define Factorization-Machine based Neural Network Model"""

    def _build_graph(self, hparams):
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        with tf.variable_scope("CCCFModel") as scope:
            with tf.variable_scope("cf_user_embedding", initializer=self.initializer) as escope:
                self.cf_user_embedding = tf.get_variable(name='cf_user_embedding_layer',
                                                         shape=[hparams.n_user, hparams.dim],
                                                         dtype=tf.float32)
                self.embed_params.append(self.cf_user_embedding)
                self.cf_user_embedding_bias = tf.concat(
                    [self.cf_user_embedding, tf.ones((hparams.n_user, 1), dtype=tf.float32)], 1,
                    name='cf_user_embedding_bias')

            with tf.variable_scope("cf_item_embedding", initializer=self.initializer) as escope:
                self.cf_item_embedding = tf.get_variable(name='cf_item_embedding_layer',
                                                         shape=[hparams.n_item, hparams.dim],
                                                         dtype=tf.float32)
                self.embed_params.append(self.cf_item_embedding)
                self.cf_item_embedding_bias = tf.concat(
                    [tf.ones((hparams.n_item, 1), dtype=tf.float32), self.cf_item_embedding], 1,
                    name='item_cf_embedding_bias')

            with tf.variable_scope("user_attr_embedding", initializer=self.initializer) as escope:
                self.user_attr_embedding = tf.get_variable(name='user_attr_embedding_layer',
                                                           shape=[hparams.n_user_attr, hparams.dim],
                                                           dtype=tf.float32)
                self.embed_params.append(self.user_attr_embedding)

            with tf.variable_scope("item_attr_embedding", initializer=self.initializer) as escope:
                self.item_attr_embedding = tf.get_variable(name='item_attr_embedding_layer',
                                                           shape=[hparams.n_item_attr, hparams.dim],
                                                           dtype=tf.float32)
                self.embed_params.append(self.item_attr_embedding)

            cf_score = self._build_cf_embedding_score(hparams)
            factor_score = self._build_factor_embedding_score(hparams)
            pred = cf_score + factor_score
            return pred

    def _build_cf_embedding_score(self, hparams):
        with tf.variable_scope("cf_embedding_score", initializer=self.initializer) as scope:
            cf_user = tf.nn.embedding_lookup(self.cf_user_embedding_bias, self.iterator.userIds,
                                             name='cf_user_embedding')
            cf_item = tf.nn.embedding_lookup(self.cf_item_embedding_bias, self.iterator.itemIds,
                                             name='cf_item_embedding')
            cf_score = tf.reduce_sum(tf.multiply(cf_user, cf_item), 1) + hparams.mu
            return cf_score

    def _build_factor_embedding_score(self, hparams):
        with tf.variable_scope("factor_user", initializer=self.initializer) as scope:
            factor_user = self._build_dnn_embedding(self.user_attr_embedding, self.iterator.user_profiles_indices,
                                                    self.iterator.user_profiles_values, \
                                                    self.iterator.user_profiles_weights,
                                                    self.iterator.user_profiles_shape, hparams)
        with tf.variable_scope("factor_item", initializer=self.initializer) as scope:
            factor_item = self._build_dnn_embedding(self.item_attr_embedding, self.iterator.item_profiles_indices,
                                                    self.iterator.item_profiles_values, \
                                                    self.iterator.item_profiles_weights,
                                                    self.iterator.item_profiles_shape, hparams)
        factor_score = tf.reduce_sum(tf.multiply(factor_user, factor_item), 1)
        return factor_score

    def _build_dnn_embedding(self, embedding, dnn_indices, dnn_values, dnn_weights, dnn_shape, hparams):
        dnn_sparse_index = tf.SparseTensor(dnn_indices, dnn_values, dnn_shape)
        dnn_sparse_weight = tf.SparseTensor(dnn_indices, dnn_weights, dnn_shape)
        dnn_input_orgin = tf.nn.embedding_lookup_sparse(embedding, dnn_sparse_index, dnn_sparse_weight, combiner="sum")
        last_layer_size = hparams.dim
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(dnn_input_orgin)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(layer_idx),
                                                  shape=[last_layer_size, layer_size],
                                                  dtype=tf.float32)
                curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(layer_idx),
                                                  shape=[layer_size],
                                                  dtype=tf.float32)
                tf.summary.histogram("nn_part/" + 'w_nn_layer' + str(layer_idx),
                                     curr_w_nn_layer)
                tf.summary.histogram("nn_part/" + 'b_nn_layer' + str(layer_idx),
                                     curr_b_nn_layer)
                curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx],
                                                       curr_w_nn_layer,
                                                       curr_b_nn_layer)
                scope = "nn_part" + str(idx)
                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer,
                                                          scope=scope,
                                                          activation=activation,
                                                          layer_idx=idx)
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)
            return hidden_nn_layers[-1]
