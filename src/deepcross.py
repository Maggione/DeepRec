"""define Factorization-Machine based Neural Network Model"""
import math
import numpy as np
import tensorflow as tf
from src.base_model import BaseModel

__all__ = ["DeepCrossModel"]


class DeepCrossModel(BaseModel):
    """define Factorization-Machine based Neural Network Model"""

    def _build_graph(self, hparams):
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        with tf.variable_scope("DeepCross") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(name='embedding_layer',
                                                 shape=[hparams.FEATURE_COUNT, hparams.dim],
                                                 dtype=tf.float32)
                self.embed_params.append(self.embedding)
                embed_out, embed_layer_size = self._build_embedding(hparams)
            logit = self._build_linear(hparams)
            # logit = tf.add(logit, self._build_fm(hparams))
            logit = tf.add(logit, self._build_cross(hparams, embed_out))
            logit = tf.add(logit, self._build_dnn(hparams, embed_out, embed_layer_size))
            return logit

    def _build_embedding(self, hparams):
        fm_sparse_index = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                          self.iterator.dnn_feat_values,
                                          self.iterator.dnn_feat_shape)
        fm_sparse_weight = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_weights,
                                           self.iterator.dnn_feat_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(self.embedding,
                                                            fm_sparse_index,
                                                            fm_sparse_weight,
                                                            combiner="sum")
        embedding = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT])
        embedding_size = hparams.FIELD_COUNT * hparams.dim
        return embedding, embedding_size

    def _build_linear(self, hparams):
        with tf.variable_scope("linear_part", initializer=self.initializer) as scope:
            w_linear = tf.get_variable(name='w',
                                       shape=[hparams.FEATURE_COUNT, 1],
                                       dtype=tf.float32)
            b_linear = tf.get_variable(name='b',
                                       shape=[1],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
            x = tf.SparseTensor(self.iterator.fm_feat_indices,
                                self.iterator.fm_feat_values,
                                self.iterator.fm_feat_shape)
            linear_output = tf.add(tf.sparse_tensor_dense_matmul(x, w_linear), b_linear)
            self.layer_params.append(w_linear)
            self.layer_params.append(b_linear)
            tf.summary.histogram("linear_part/w", w_linear)
            tf.summary.histogram("linear_part/b", b_linear)
            return linear_output

    def _build_cross(self, hparams, nn_input):
        hidden_nn_layers = []
        # nn_input = tf.matrix_transpose(nn_input)
        hidden_nn_layers.append(nn_input)
        dim_size = nn_input.get_shape()[-1]
        with tf.variable_scope("cross_part", initializer=self.initializer) as scope:
            for idx in range(hparams.cross_layers):
                curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(idx),
                                                  shape=[dim_size, 1],
                                                  dtype=tf.float32)
                curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(idx),
                                                  shape=[dim_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
                cross_feature = tf.matmul(hidden_nn_layers[-1], curr_w_nn_layer)
                cross_feature = tf.tile(cross_feature, [1, dim_size])
                cross_feature = tf.multiply(cross_feature, hidden_nn_layers[0])
                curr_hidden_nn_layer = tf.nn.bias_add(cross_feature, curr_b_nn_layer)
                curr_hidden_nn_layer = tf.add(curr_hidden_nn_layer, hidden_nn_layers[-1])
                hidden_nn_layers.append(curr_hidden_nn_layer)

                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)
            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[dim_size, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32)
            cross_out = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)

            return cross_out


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


    def _build_dnn(self, hparams, embed_out, embed_layer_size):
        """
        fm_sparse_index = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                          self.iterator.dnn_feat_values,
                                          self.iterator.dnn_feat_shape)
        fm_sparse_weight = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_weights,
                                           self.iterator.dnn_feat_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(self.embedding,
                                                            fm_sparse_index,
                                                            fm_sparse_weight,
                                                            combiner="sum")
        w_fm_nn_input = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT])
        last_layer_size = hparams.FIELD_COUNT * hparams.dim
        """
        w_fm_nn_input = embed_out
        last_layer_size = embed_layer_size
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(layer_idx),
                                                  shape=[last_layer_size, layer_size],
                                                  dtype=tf.float32)
                curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(layer_idx),
                                                  shape=[layer_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
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

            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[last_layer_size, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            tf.summary.histogram("nn_part/" + 'w_nn_output' + str(layer_idx),
                                 w_nn_output)
            tf.summary.histogram("nn_part/" + 'b_nn_output' + str(layer_idx),
                                 b_nn_output)
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            return nn_output
