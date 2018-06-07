"""define inner product neural network model"""
import math
import tensorflow as tf
import numpy as np
from src.base_model import BaseModel

__all__ = ["IpnnModel"]


class IpnnModel(BaseModel):
    """define product neural network model"""

    def _build_graph(self, hparams):
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        with tf.variable_scope("Ipnn") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(name='embedding_layer',
                                                 shape=[hparams.FEATURE_COUNT, hparams.dim],
                                                 dtype=tf.float32)
                self.embed_params.append(self.embedding)
            logit = self._build_pnn(hparams)
        return logit

    def _build_product_layer(self, hparams):
        num_inputs = hparams.FIELD_COUNT
        embed_size = hparams.dim
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        node_in = num_inputs * hparams.dim + num_pairs
        fm_sparse_indexs = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_values,
                                           self.iterator.dnn_feat_shape)
        fm_sparse_weight = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_weights,
                                           self.iterator.dnn_feat_shape)
        xw = tf.nn.embedding_lookup_sparse(self.embedding,
                                           fm_sparse_indexs,
                                           fm_sparse_weight,
                                           combiner="sum")
        xw = tf.reshape(xw, [-1, num_inputs * embed_size])
        xw3d = tf.reshape(xw, [-1, num_inputs, embed_size])
        row = []
        col = []
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        # batch * pair * k
        p = tf.transpose(
            # pair * batch * k
            tf.gather(
                # num * batch * k
                tf.transpose(
                    xw3d, [1, 0, 2]),
                row),
            [1, 0, 2])
        # batch * pair * k
        q = tf.transpose(
            tf.gather(
                tf.transpose(
                    xw3d, [1, 0, 2]),
                col),
            [1, 0, 2])
        p = tf.reshape(p, [-1, num_pairs, embed_size])
        q = tf.reshape(q, [-1, num_pairs, embed_size])
        ip = tf.reshape(tf.reduce_sum(p * q, [-1]), [-1, num_pairs])
        l = tf.concat([xw, ip], 1)
        return l, node_in

    def _build_pnn(self, hparams):
        w_fm_nn_input, last_layer_size = self._build_product_layer(hparams)
        # print(w_fm_nn_input)
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
                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer, scope=scope,
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
            b_nn_output = tf.get_variable(name='b_nn_output', shape=[1], dtype=tf.float32)
            tf.summary.histogram("nn_part/" + 'w_nn_output' + str(layer_idx), w_nn_output)
            tf.summary.histogram("nn_part/" + 'b_nn_output' + str(layer_idx), b_nn_output)
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            return nn_output
