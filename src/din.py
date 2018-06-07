"""define deep interest network model"""
import math
import tensorflow as tf
import numpy as np
from src.base_model import BaseModel

__all__ = ["DinModel"]


# reference: Deep Interest Network for Click-Through Rate Prediction
# dinModel = deepM + attention mechanism

class DinModel(BaseModel):
    """define deep interest network model"""

    def _build_graph(self, hparams):
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        with tf.variable_scope("Din") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(name='embedding_layer',
                                                 shape=[hparams.FEATURE_COUNT, hparams.dim],
                                                 dtype=tf.float32)
                self.embed_params.append(self.embedding)
            logit = self._build_linear(hparams)
            logit = tf.add(logit, self._build_fm(hparams))
            logit = tf.add(logit, self._build_din(hparams))
            return logit

    # 将不需要构建成attention field pair，使用embedding_lookup_sparse去获得第一层的embedding输入, 可以认为是DNN 模块
    def _build_dnn_embedding(self, hparams):
        dim = hparams.dim
        DNN_FIELD_NUM = hparams.DNN_FIELD_NUM
        dnn_feat_indexs = tf.SparseTensor(self.iterator.dnn_feat_indices, \
                                          self.iterator.dnn_feat_values, \
                                          self.iterator.dnn_feat_shape)
        dnn_feat_weight = tf.SparseTensor(self.iterator.dnn_feat_indices, \
                                          self.iterator.dnn_feat_weight, \
                                          self.iterator.dnn_feat_shape)
        dnn_embedding_output = tf.reshape(
            tf.nn.embedding_lookup_sparse(self.embedding, dnn_feat_indexs, \
                                          dnn_feat_weight, combiner="sum"), \
            [-1, dim * DNN_FIELD_NUM])
        return dnn_embedding_output

    # attention model 是pair_to_pair的形式,对于指定的field去构建一个attention model
    # user_input_indices:用户稀疏输入矩阵的indices
    # user_input_shape:用户输入矩阵的shape
    # user_feat_index_sparse:稀疏矩阵,存储用户每个field的feat编号
    # user_feat_weight_sparse:稀疏矩阵,存储用户每个field的取值,也可以作为weight
    # 对单pair特征构建attention单元
    def _build_attention_pair(self, user_input_indices, user_input_shape,
                              user_feat_index_sparse, user_feat_weight_sparse,
                              news_embedding_batch_dense, w_embedding_dense,
                              newtwork_id, hparams):

        batch_size = hparams.batch_size
        dim = hparams.dim
        attention_module_params = []

        split_feat_index = tf.sparse_split(axis=0, num_split=batch_size, \
                                           sp_input=user_feat_index_sparse)
        # attention的网络参数必须是共享的
        with tf.variable_scope("attention_net_" + str(newtwork_id), \
                               initializer=self.initializer) as scope:
            weight_temp = []
            # 只能对batch中的每个样本进行循环
            for index, feat_index in enumerate(split_feat_index):
                # 提取news_embedding中,batch_size中的一行
                news_embedding = tf.gather(news_embedding_batch_dense, index)
                # 将user field所对应的feat的embedding取出来
                # attention网络的输入包含三个部分,news特征向量,news特征向量和user特征向量的差,以及user特征向量本身
                feat_embeding = tf.nn.embedding_lookup(w_embedding_dense, feat_index.values)
                sub_embedding = tf.subtract(feat_embeding, news_embedding)
                news_embedding_repeat = tf.add(tf.zeros_like(feat_embeding), news_embedding)
                attention_x = tf.concat(axis=1, values= \
                    [feat_embeding, sub_embedding, news_embedding_repeat])
                # 网络参数必须是共享的
                # attention网络的隐层结点的个数设置为embedding的维度
                # attention网络的输入包括的user embedding和item embedding和(user embedding-item embedding)
                # 因此attention网络的输入维度=3*dim
                attention_hidden_sizes = hparams.attention_layer_sizes
                attention_w = tf.get_variable(name="attention_hidden_w", \
                                              shape=[dim * 3, attention_hidden_sizes], \
                                              dtype=tf.float32)

                attention_b = tf.get_variable(name="attention_hidden_b", \
                                              shape=[attention_hidden_sizes], \
                                              dtype=tf.float32)
                curr_attention_layer = tf.nn.xw_plus_b(attention_x, attention_w, attention_b)
                curr_attention_layer = self._activate(logit=curr_attention_layer,
                                                      activation=hparams.attention_activation)

                attention_output_w = tf.get_variable(name="attention_output_w", \
                                                     shape=[attention_hidden_sizes, 1], \
                                                     dtype=tf.float32)
                attention_output_b = tf.get_variable(name="attention_output_b", \
                                                     shape=[1], \
                                                     dtype=tf.float32)
                attention_output = tf.nn.sigmoid(
                    tf.nn.xw_plus_b(curr_attention_layer, \
                                    attention_output_w, \
                                    attention_output_b))
                weight_temp.append(attention_output)
                if index == 0:
                    attention_module_params.append(attention_w)
                    attention_module_params.append(attention_b)
                    attention_module_params.append(attention_output_w)
                    attention_module_params.append(attention_output_b)
                    tf.summary.histogram("attention_w", attention_w)
                    tf.summary.histogram("attention_b", attention_b)
                    tf.summary.histogram("attention_output_w", attention_output_w)
                    tf.summary.histogram("attention_output_b", attention_output_b)

                scope.reuse_variables()

            attention_weight = tf.reshape(tf.concat(weight_temp, 0), [-1])

            sparse_attention_weight = tf.SparseTensor(indices=user_input_indices, \
                                                      values=attention_weight, \
                                                      dense_shape=user_input_shape)

            # 转换为dense tensor方便相乘,然后再转换到sparse tensor
            user_feat_weight_dense = tf.sparse_tensor_to_dense(user_feat_weight_sparse)
            attention_weight_dense = tf.sparse_tensor_to_dense(sparse_attention_weight)

            final_weight = tf.multiply(user_feat_weight_dense, attention_weight_dense)
            # 只提取not equal 0的是不严谨的,很容易出现bug.神经网络的激活值很容易出现等于0
            # final_weight_val = tf.gather_nd(final_weight, tf.where(tf.not_equal(final_weight,0)))
            embedding_sparse_weights = tf.SparseTensor(indices=user_input_indices, \
                                                       values=tf.gather_nd(final_weight, \
                                                                           user_input_indices), \
                                                       dense_shape=user_input_shape)

            final_embedding = tf.nn.embedding_lookup_sparse(w_embedding_dense, \
                                                            user_feat_index_sparse, \
                                                            embedding_sparse_weights, \
                                                            combiner="sum")
            attention_embedding = tf.reshape(final_embedding, [-1, dim])
            return attention_embedding, attention_module_params

    # 对pair特征构建attention单元，对于pair的field使用attention机制去获得embedding.
    def _build_attention_embedding(self, hparams):

        PAIR_NUM = hparams.PAIR_NUM
        batch_size = hparams.batch_size
        # sparse input
        # news pair的sparse表示,shape=[attention_num*batch_size, feature_num]
        attention_news_x = tf.SparseTensor(
            indices=self.iterator.attention_news_indices,
            values=self.iterator.attention_news_values,
            dense_shape=self.iterator.attention_news_shape)
        #
        attention_user_feat_index_batch = tf.SparseTensor(
            indices=self.iterator.attention_user_indices,
            values=self.iterator.attention_user_values,
            dense_shape=self.iterator.attention_user_shape)
        attention_user_feat_weight_batch = tf.SparseTensor(
            indices=self.iterator.attention_user_indices,
            values=self.iterator.attention_user_weights,
            dense_shape=self.iterator.attention_user_shape)

        attention_news_x_embedding = tf.sparse_tensor_dense_matmul(attention_news_x, self.embedding)
        split_attention_x_embedding = tf.split(attention_news_x_embedding,
                                               num_or_size_splits=PAIR_NUM * batch_size,
                                               axis=0)

        # 将news_embedding分成PAIR_NUM份
        # 每份都代表一个field的embedding,size = [batch_size, embedding_dim]
        attention_news_embedding_pair = []
        for pair_idx in range(PAIR_NUM):
            tmp = []
            for i in range(batch_size):
                tmp.append(split_attention_x_embedding[i * PAIR_NUM + pair_idx])
            attention_news_embedding_pair.append(tf.concat(tmp, axis=0))

        split_attention_user_feat_index_batch = tf.sparse_split(
            axis=0,
            num_split=PAIR_NUM * batch_size,
            sp_input=attention_user_feat_index_batch)
        split_attention_user_feat_weight_batch = tf.sparse_split(
            axis=0,
            num_split=PAIR_NUM * batch_size,
            sp_input=attention_user_feat_weight_batch)

        attention_user_feat_index_batch_pair = []
        attention_user_feat_weight_batch_pair = []
        for pair_idx in range(PAIR_NUM):
            tmp1 = []
            tmp2 = []
            for i in range(batch_size):
                tmp1.append(split_attention_user_feat_index_batch[i * PAIR_NUM + pair_idx])
                tmp2.append(split_attention_user_feat_weight_batch[i * PAIR_NUM + pair_idx])
            attention_user_feat_index_batch_pair.append(tf.sparse_concat(axis=0, sp_inputs=tmp1))
            attention_user_feat_weight_batch_pair.append(tf.sparse_concat(axis=0, sp_inputs=tmp2))
        # #news embedding
        nn_input = []
        attention_nn_params = []
        for idx in range(PAIR_NUM):
            attention_embedding, attention_module_params = self._build_attention_pair(
                attention_user_feat_index_batch_pair[idx].indices,
                attention_user_feat_index_batch_pair[idx].dense_shape,
                attention_user_feat_index_batch_pair[idx],
                attention_user_feat_weight_batch_pair[idx],
                attention_news_embedding_pair[idx],
                self.embedding, idx, hparams)
            field_nn_input = tf.concat([attention_embedding, attention_news_embedding_pair[idx]], 1)
            nn_input.append(field_nn_input)
            attention_nn_params.extend(attention_module_params)
        #
        for param in attention_nn_params:
            self.layer_params.append(param)

        concat_nn_input = tf.concat(nn_input, 1)
        return concat_nn_input, attention_nn_params

    # build DIN model = fm part + linear part + attention part + dnn part
    def _build_din(self, hparams):

        PAIR_NUM = hparams.PAIR_NUM
        DNN_FIELD_NUM = hparams.DNN_FIELD_NUM
        # batch_size = hparams.batch_size
        dim = hparams.dim
        FEATURE_COUNT = hparams.FEATURE_COUNT
        layer_sizes = hparams.layer_sizes

        # build attention module
        # 返回的attention embedding是包含user 和 news attention embedding
        # attention_embedding_output shape = [batch_size, dim*PAIR_NUM*2]
        attention_embedding_output, attention_nn_params = self._build_attention_embedding(hparams)
        # print('attention_nn_params tensor num:', len(attention_nn_params))
        # build dnn module
        # dnn embedding shape = [batch_size, DNN_FIELD_NUM*dim]
        dnn_embedding_output = self._build_dnn_embedding(hparams)
        # final input for DNN
        nn_input = tf.concat([attention_embedding_output, dnn_embedding_output], 1)
        last_layer_size = (PAIR_NUM * 2 + DNN_FIELD_NUM) * dim
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(nn_input)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(layer_sizes):
                curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(layer_idx), \
                                                  shape=[last_layer_size, layer_size], \
                                                  dtype=tf.float32)
                curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(layer_idx), \
                                                  shape=[layer_size], \
                                                  dtype=tf.float32)
                tf.summary.histogram("nn_part/" + 'w_nn_layer' + str(layer_idx), curr_w_nn_layer)
                tf.summary.histogram("nn_part/" + 'b_nn_layer' + str(layer_idx), curr_b_nn_layer)
                curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx], \
                                                       curr_w_nn_layer, \
                                                       curr_b_nn_layer)
                scope = "nn_part" + str(idx)
                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer,
                                                          scope=scope,
                                                          activation=activation, \
                                                          layer_idx=idx)
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)

            w_nn_output = tf.get_variable(name='w_nn_output', \
                                          shape=[last_layer_size, 1], \
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output', shape=[1], dtype=tf.float32)
            tf.summary.histogram("nn_part/" + 'w_nn_output' + str(layer_idx), w_nn_output)
            tf.summary.histogram("nn_part/" + 'b_nn_output' + str(layer_idx), b_nn_output)
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            return nn_output

    # build linear part
    def _build_linear(self, hparams):
        with tf.variable_scope("linear_part", initializer=self.initializer) as scope:
            FEATURE_COUNT = hparams.FEATURE_COUNT
            w_linear = tf.get_variable(name='w', shape=[FEATURE_COUNT, 1], dtype=tf.float32)
            b_linear = tf.get_variable(name='b', shape=[1], dtype=tf.float32)
            x_input = tf.SparseTensor(self.iterator.fm_feat_indices, \
                                      self.iterator.fm_feat_val, \
                                      self.iterator.fm_feat_shape)
            linear_output = tf.add(tf.sparse_tensor_dense_matmul(x_input, w_linear), b_linear)
            return linear_output

    # build fm part
    def _build_fm(self, hparams):
        with tf.variable_scope("fm_part") as scope:
            x_input = tf.SparseTensor(self.iterator.fm_feat_indices, \
                                      self.iterator.fm_feat_val, \
                                      self.iterator.fm_feat_shape)
            xx_input = tf.SparseTensor(self.iterator.fm_feat_indices, \
                                       tf.pow(self.iterator.fm_feat_val, 2), \
                                       self.iterator.fm_feat_shape)
            fm_output = 0.5 * tf.reduce_sum(
                tf.pow(tf.sparse_tensor_dense_matmul(x_input, self.embedding), 2) - \
                tf.sparse_tensor_dense_matmul(xx_input, \
                                              tf.pow(self.embedding, 2)), \
                1, \
                keep_dims=True)
            return fm_output
