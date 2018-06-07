import math
import tensorflow as tf
import numpy as np
from src.base_model import BaseModel

__all__ = ["RippleModel"]


class RippleModel(BaseModel):
    def _build_graph(self, config):
        self.n_entity = config.n_entity  # number of  entity
        self.n_relation = config.n_relation  # number of relation
        self.n_entity_emb = config.n_entity_emb  # the dimension of the entity,
        self.n_relation_emb = config.n_relation_emb  # the dimension of the realtion
        self.n_map_emb = config.n_map_emb  # the dimension after mapping entity to a new space
        self.n_memory = config.n_memory  # the size of memory
        self.n_hops = config.n_hops  # the number of hops
        self.reg_kg = config.reg_kg
        if config.dtype == 16:
            self.dtype = tf.float16
        elif config.dtype == 32:
            self.dtype = tf.float32
        elif config.dtype == 64:
            self.dtype = tf.float64
        else:
            self.dtype = tf.float32
        # self.dtype = config.dtype  # data type, [tf.float16, tf.float32, tf.float64]
        self.predict_mode = config.predict_mode  # different method to do the final prediction
        self.n_DCN_layer = config.n_DCN_layer  # when choose DCN to do the prediction, the number of DCN layers
        self.output_using_all_hops = config.output_using_all_hops  # bool, if using all hops o when do the prediction
        self.item_update_mode = config.item_update_mode  # different wat to update the item embedding
        self.is_map_feature = config.is_map_feature
        self.is_use_relation = config.is_use_relation
        self.keep_prob_train = 1 - np.array(config.dropout)
        self.keep_prob_test = np.ones_like(config.dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        self.hparams = config
        logits = self._build_model()
        return logits

    def _get_loss(self, hparams):
        # cross entropy loss
        xe_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.iterator.labels, logits=self.logit))
        # knowledge graph loss
        score_sum = 0
        for hop in range(self.n_hops):
            expand_h = tf.expand_dims(self.h_emb_hops[hop], axis=2)
            expand_t = tf.expand_dims(self.t_emb_hops[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(expand_h, self.r_emb_hops[hop]), expand_t))
            zero_mask = (self.mem_mask[hop] + 1e6) / 1e6
            score_sum += tf.reduce_mean(tf.sigmoid(hRt) * zero_mask, axis=[0,1])
        kg_loss = - hparams.reg_kg * score_sum
        self.data_loss = xe_loss + kg_loss

        # embedding loss
        emb_reg_loss = 0
        for hop in range(self.n_hops):
            zero_mask = (self.mem_mask[hop] + 1e6) / 1e6
            emb_reg_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_hops[hop] * self.h_emb_hops[hop], axis=2) * zero_mask, axis=[0,1])
            emb_reg_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_hops[hop] * self.t_emb_hops[hop], axis=2) * zero_mask, axis=[0,1])
            emb_reg_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_hops[hop] * self.r_emb_hops[hop], axis=[2,3]) * zero_mask, axis=[0,1])
        emb_reg_loss = hparams.embed_l2 * emb_reg_loss
        self.regular_loss = emb_reg_loss

        self.loss = xe_loss + kg_loss + emb_reg_loss
        return self.loss



    def _build_model(self):
        with tf.variable_scope("Ripple", initializer=self.initializer) as scope:
            self.build_embeddings()
            # feature map matrix
            self.A = tf.get_variable(name="map_A", shape=[self.n_entity_emb, self.n_map_emb], dtype=self.dtype)
            # prediction weight for different hops
            self.w_predict = []
            for i in range(self.n_hops + 1):
                self.w_predict.append(tf.get_variable(name="w_predict" + str(i), shape=[1],
                                             dtype=self.dtype,initializer=tf.constant_initializer(1.0)))
            # item retrival
            # (batch size, n_entity_emb)
            item_emb = tf.nn.embedding_lookup(self.entity_embedding, self.iterator.item)
            if self.is_map_feature:
                item_emb_map = self._active_layer(tf.matmul(item_emb, self.A), scope, activation[0])
            else:
                item_emb_map = item_emb
            # user key retrieval
            self.h_emb_hops = []
            self.r_emb_hops = []
            self.t_emb_hops = []
            self.mem_h = tf.unstack(self.iterator.mem_h)
            self.mem_r = tf.unstack(self.iterator.mem_r)
            self.mem_t = tf.unstack(self.iterator.mem_t)
            self.mem_len = tf.unstack(self.iterator.mem_len)
            self.mem_mask = tf.sequence_mask(self.mem_len, self.hparams.n_memory)
            self.mem_mask = (tf.cast(self.mem_mask, tf.float32) - 1) * (-1)
            self.mem_mask = self.mem_mask * -1e6
            # self.mem_mask = tf.unstack(self.iterator.mem_mask)

            for i in range(self.n_hops):
                # (batch size, n_memory, n_entity_emb)
                self.h_emb_hops.append(tf.nn.embedding_lookup(self.entity_embedding, self.mem_h[i]))
                # (batch size, n_memory, n_relation_emb, n_relation_emb)
                if self.is_use_relation:
                    self.r_emb_hops.append(tf.nn.embedding_lookup(self.relation_embedding, self.mem_r[i]))
                # (batch size, n_memory, n_entity_emb)
                self.t_emb_hops.append(tf.nn.embedding_lookup(self.entity_embedding, self.mem_t[i]))
            # two transformation matrix for when updating item embeddings
            self.o_map_mat = tf.get_variable("o_map_mat", shape=[self.n_map_emb, self.n_map_emb], dtype=self.dtype)
            self.item_map_mat = tf.get_variable("item_map_mat", shape=[self.n_map_emb, self.n_map_emb], dtype=self.dtype)
            # final output of the key_addressing
            o_list = self.key_addressing(item_emb_map, self.h_emb_hops, self.r_emb_hops, self.t_emb_hops)
            # make prediction
            self.o = o_list[-1]
            logits = tf.squeeze(self.make_prediction(item_emb_map, o_list))
            return  logits

    def build_embeddings(self):
        with tf.name_scope("embedding"):
            self.entity_embedding = tf.get_variable(name="entity", dtype=self.dtype,
                                                     shape=[self.n_entity, self.n_entity_emb])
            if self.is_use_relation:
                self.relation_embedding = tf.get_variable(name="relation", dtype=self.dtype,
                                                           shape=[self.n_relation, self.n_relation_emb,
                                                                  self.n_relation_emb])

    def key_addressing(self, item_emb_map, h_emb_hops, r_emb_hops, t_emb_hops):
        """

        :param item_emb_map: [batch_size, n_entity_emb]
        :param h_emp_hops: list, whose element shape [batch_size, n_memory, n_entity_emb]
        :param t_emp_hops: list,
        :return:
        """
        if self.is_map_feature:
            A = tf.tile(tf.expand_dims(self.A, axis=0), multiples=[tf.shape(item_emb_map)[0], 1, 1])
        o_list = []
        for hop in range(self.n_hops):
            if self.is_use_relation:
                #  h_emb_hops[hop]: [batch_size, n_memory, n_entity_emb] - > [batch_size, n_memory, n_entity, 1]
                expand_h = tf.expand_dims(h_emb_hops[hop], axis=3)
                # R: [batch_size, n_memory, n_relation_emb, n_relation_emb],
                # n_relation_emb = n_entity_emb
                # Rh : [batch_size, n_memory, n_entity_emb]
                Rh = tf.squeeze(tf.matmul(r_emb_hops[hop], expand_h), axis=3)
            else:
                Rh = h_emb_hops[hop]
            # if do the feature map or not, multiply A
            # if map, Rh_map : [batch_size, n_memory, n_map_emb]
            # if map, item_emb_map: [batch_size, n_map_emb]
            if self.is_map_feature:
                Rh_map = self.act_func(tf.matmul(Rh, A))
            else:
                Rh_map = Rh
            # Softmax
            expand_item_emb = tf.expand_dims(item_emb_map, axis=2)  # [batch_size, n_map_emb, 1]
            dotted = tf.squeeze(tf.matmul(Rh_map, expand_item_emb), axis=2)  # [batch_size, n_memory]
            # get off the effect of null paddings
            soft_dotted = dotted - self.mem_mask[hop]
            # calculate probabilities
            probs = tf.nn.softmax(soft_dotted)  # [batch_size, n_memory]
            # t: [batch_size, n_memory, n_entity_emb]
            # t_map: [batch_size, n_memory, n_map_emb]
            if self.is_map_feature:
                t_map = self.act_func(tf.matmul(t_emb_hops[hop], A))
            else:
                t_map = t_emb_hops[hop]
            # prob: [batch_size, n_memory] -> [batch_size, n_memory, 1]
            expand_probs = tf.expand_dims(probs, axis=2)
            # o: [batch_size, n_entity_emb]
            o = tf.reduce_sum(t_map * expand_probs, axis=1)
            # update the item embedding
            item_emb_map = self.update_item_embedding(item_emb_map, o)
            o_list.append(o)

        return o_list


    def make_prediction(self, item_emb, o_list):
        """make prediction according to the item embedding and the variable o
        :return:
        """

        if self.predict_mode == "inner_product":
            return self._inner_product_predict(item_emb, o_list)
        elif self.predict_mode == "MLP":
            return self._MLP_predict(item_emb, o_list)
        elif self.predict_mode == "DCN":
            return self._DCN_predict(item_emb, o_list)

    def _inner_product_predict(self, x, o_list):
        """compute  the probability according to the inner product
        :param x:
        :param y:
        :return:
        """

        y = o_list[-1] * self.w_predict[-1]
        if self.output_using_all_hops:
            for i in range(self.n_hops-1):
                y += o_list[i] * self.w_predict[i]
        logits = tf.reduce_sum(x * y, axis=1)
        return logits

    def _MLP_predict(self, x, o_list):
        """
        :param x: [batch_size, n_entity_emb], variabale item_emb
        :param y: [batch_size, n_entity_emb], input o
        :return prob: [batch_size]
        """

        y = o_list[-1]
        # [batch_size, 2*n_entity_emb]
        out_feature = tf.concat([x, y], axis=0)
        w_pred = tf.get_variable("w_pred", shape=[2 * self.n_map_emb, 1], dtype=self.dtype,)
        b_pred = tf.get_variable("b_pred", shape=[1], dtype=self.dtype, initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(out_feature, w_pred) + b_pred

        return logits

    def _DCN_predict(self, item_emb, o_list):
        """Deep & Cross network,
        ref: Deep & Cross Network for Ad Click Predictions
        :return:
        """

        if self.output_using_all_hops:
            x0 = tf.concat([item_emb] + o_list, axis=1)  # [batch_size, (n_hops + 1)*n_map_emb]
            x_dim = (self.n_hops + 1) * self.n_map_emb
        else:
            x0 = tf.concat([item_emb, o_list[-1]], axis=1)  #[batch_size, 2*n_map_emb]
            x_dim = 2 * self.n_map_emb
        x0 = tf.expand_dims(x0, axis=2)  # [batch_size, 2*n_map_emb, 1]
        output_layer = [x0]
        for i in range(self.n_DCN_layer):
            w = tf.get_variable("w_dcn_" + str(i), shape=[x_dim, 1],dtype=self.dtype, initializer=tf.contrib.layers.xavier_initializer())
            w_tile = tf.tile(tf.expand_dims(w, axis=0), multiples=[tf.shape(item_emb)[0], 1, 1])
            b = tf.get_variable("b_dcn_" + str(i), shape=[x_dim, 1], dtype=self.dtype, initializer=tf.constant_initializer(0.0))
            output_layer.append(tf.matmul(tf.matmul(x0, output_layer[-1], transpose_b=True), w_tile) + b + output_layer[-1])
        # final layer
        w = tf.get_variable("w_pred", shape=[x_dim, 1], dtype=self.dtype, initializer=tf.contrib.layers.xavier_initializer())
        logits = tf.matmul(tf.squeeze(output_layer[-1]), w)
        return logits

    def update_item_embedding(self, item_emb, o):
        """update the item_emb

        different ways to update
        """

        if self.item_update_mode == "plus":
            item_emb = item_emb + o
        elif self.item_update_mode == "map_o":
            item_emb = item_emb + tf.matmul(o, self.o_map_mat)
        elif self.item_update_mode == "map_item":
            item_emb = tf.matmul(item_emb + o, self.item_map_mat)
        elif self.item_update_mode == "map_all":
            item_emb = tf.matmul(item_emb + tf.matmul(o, self.o_map_mat), self.item_map_mat)

        return item_emb