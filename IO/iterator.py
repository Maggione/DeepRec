"""define iterator"""
import collections
import tensorflow as tf
import abc
import numpy as np

BUFFER_SIZE = 256
__all__ = ["BaseIterator", "FfmIterator", "DinIterator", "CCCFNetIterator", "DKNIterator", "MKRIterator", "RippleIterator"]


class BaseIterator(object):
    @abc.abstractmethod
    def get_iterator(self, src_dataset):
        """Subclass must implement this."""
        pass

    @abc.abstractmethod
    def parser(self, record):
        pass


class FfmIterator(BaseIterator):
    def __init__(self, src_dataset):
        self.get_iterator(src_dataset)

    def get_iterator(self, src_dataset):
        src_dataset = src_dataset.map(self.parser)
        # src_dataset = src_dataset.shuffle(buffer_size=BUFFER_SIZE)
        iterator = src_dataset.make_initializable_iterator()
        _fm_feat_indices, _fm_feat_values, \
        _fm_feat_shape, _labels, _dnn_feat_indices, \
        _dnn_feat_values, _dnn_feat_weights, _dnn_feat_shape = iterator.get_next()
        self.initializer = iterator.initializer
        self.fm_feat_indices = _fm_feat_indices
        self.fm_feat_values = _fm_feat_values
        self.fm_feat_shape = _fm_feat_shape
        self.labels = _labels
        self.dnn_feat_indices = _dnn_feat_indices
        self.dnn_feat_values = _dnn_feat_values
        self.dnn_feat_weights = _dnn_feat_weights
        self.dnn_feat_shape = _dnn_feat_shape

    def parser(self, record):
        keys_to_features = {
            'fm_feat_indices': tf.FixedLenFeature([], tf.string),
            'fm_feat_values': tf.VarLenFeature(tf.float32),
            'fm_feat_shape': tf.FixedLenFeature([2], tf.int64),
            'labels': tf.FixedLenFeature([], tf.string),
            'dnn_feat_indices': tf.FixedLenFeature([], tf.string),
            'dnn_feat_values': tf.VarLenFeature(tf.int64),
            'dnn_feat_weights': tf.VarLenFeature(tf.float32),
            'dnn_feat_shape': tf.FixedLenFeature([2], tf.int64),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        fm_feat_indices = tf.reshape(tf.decode_raw(parsed['fm_feat_indices'], tf.int64), [-1, 2])
        fm_feat_values = tf.sparse_tensor_to_dense(parsed['fm_feat_values'])
        fm_feat_shape = parsed['fm_feat_shape']
        labels = tf.reshape(tf.decode_raw(parsed['labels'], tf.float32), [-1, 1])
        dnn_feat_indices = tf.reshape(tf.decode_raw(parsed['dnn_feat_indices'], tf.int64), [-1, 2])
        dnn_feat_values = tf.sparse_tensor_to_dense(parsed['dnn_feat_values'])
        dnn_feat_weights = tf.sparse_tensor_to_dense(parsed['dnn_feat_weights'])
        dnn_feat_shape = parsed['dnn_feat_shape']
        return fm_feat_indices, fm_feat_values, \
               fm_feat_shape, labels, dnn_feat_indices, \
               dnn_feat_values, dnn_feat_weights, dnn_feat_shape


class DinIterator(BaseIterator):
    def __init__(self, src_dataset):
        self.get_iterator(src_dataset)

    def get_iterator(self, src_dataset):
        src_dataset = src_dataset.map(self.parser)
        # src_dataset = src_dataset.shuffle(buffer_size=BUFFER_SIZE)
        iterator = src_dataset.make_initializable_iterator()
        output = iterator.get_next()
        (_attention_news_indices, _attention_news_values, _attention_news_shape, \
         _attention_user_indices, _attention_user_values, _attention_user_weights, \
         _attention_user_shape, _fm_feat_indices, _fm_feat_val, \
         _fm_feat_shape, _labels, _dnn_feat_indices, _dnn_feat_values, \
         _dnn_feat_weight, _dnn_feat_shape) = output
        self.initializer = iterator.initializer
        self.attention_news_indices = _attention_news_indices
        self.attention_news_values = _attention_news_values
        self.attention_news_shape = _attention_news_shape
        self.attention_user_indices = _attention_user_indices
        self.attention_user_values = _attention_user_values
        self.attention_user_weights = _attention_user_weights
        self.attention_user_shape = _attention_user_shape
        self.fm_feat_indices = _fm_feat_indices
        self.fm_feat_val = _fm_feat_val
        self.fm_feat_shape = _fm_feat_shape
        self.labels = _labels
        self.dnn_feat_indices = _dnn_feat_indices
        self.dnn_feat_values = _dnn_feat_values
        self.dnn_feat_weight = _dnn_feat_weight
        self.dnn_feat_shape = _dnn_feat_shape

    def parser(self, record):
        keys_to_features = {
            'attention_news_indices': tf.FixedLenFeature([], tf.string),
            'attention_news_values': tf.VarLenFeature(tf.float32),
            'attention_news_shape': tf.FixedLenFeature([2], tf.int64),

            'attention_user_indices': tf.FixedLenFeature([], tf.string),
            'attention_user_values': tf.VarLenFeature(tf.int64),
            'attention_user_weights': tf.VarLenFeature(tf.float32),
            'attention_user_shape': tf.FixedLenFeature([2], tf.int64),

            'fm_feat_indices': tf.FixedLenFeature([], tf.string),
            'fm_feat_val': tf.VarLenFeature(tf.float32),
            'fm_feat_shape': tf.FixedLenFeature([2], tf.int64),

            'labels': tf.FixedLenFeature([], tf.string),

            'dnn_feat_indices': tf.FixedLenFeature([], tf.string),
            'dnn_feat_values': tf.VarLenFeature(tf.int64),
            'dnn_feat_weight': tf.VarLenFeature(tf.float32),
            'dnn_feat_shape': tf.FixedLenFeature([2], tf.int64),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        attention_news_indices = tf.reshape(tf.decode_raw(parsed['attention_news_indices'], \
                                                          tf.int64), [-1, 2])
        attention_news_values = tf.sparse_tensor_to_dense(parsed['attention_news_values'])
        attention_news_shape = parsed['attention_news_shape']

        attention_user_indices = tf.reshape(tf.decode_raw(parsed['attention_user_indices'], \
                                                          tf.int64), [-1, 2])
        attention_user_values = tf.sparse_tensor_to_dense(parsed['attention_user_values'])
        attention_user_weights = tf.sparse_tensor_to_dense(parsed['attention_user_weights'])
        attention_user_shape = parsed['attention_user_shape']

        fm_feat_indices = tf.reshape(tf.decode_raw(parsed['fm_feat_indices'], \
                                                   tf.int64), [-1, 2])
        fm_feat_val = tf.sparse_tensor_to_dense(parsed['fm_feat_val'])
        fm_feat_shape = parsed['fm_feat_shape']

        labels = tf.reshape(tf.decode_raw(parsed['labels'], tf.float32), [-1, 1])

        dnn_feat_indices = tf.reshape(tf.decode_raw(parsed['dnn_feat_indices'], \
                                                    tf.int64), [-1, 2])
        dnn_feat_values = tf.sparse_tensor_to_dense(parsed['dnn_feat_values'])
        dnn_feat_weight = tf.sparse_tensor_to_dense(parsed['dnn_feat_weight'])
        dnn_feat_shape = parsed['dnn_feat_shape']
        return (attention_news_indices, attention_news_values, attention_news_shape, \
                attention_user_indices, attention_user_values, attention_user_weights, \
                attention_user_shape, fm_feat_indices, fm_feat_val, \
                fm_feat_shape, labels, dnn_feat_indices, dnn_feat_values, \
                dnn_feat_weight, dnn_feat_shape)


class CCCFNetIterator(BaseIterator):
    def __init__(self, src_dataset):
        self.get_iterator(src_dataset)

    def get_iterator(self, src_dataset):
        src_dataset = src_dataset.map(self.parser)
        # src_dataset = src_dataset.shuffle(buffer_size=BUFFER_SIZE)
        iterator = src_dataset.make_initializable_iterator()
        _labels, _userIds, _itemIds, \
        _user_profiles_indices, _user_profiles_values, _user_profiles_weights, _user_profiles_shape, \
        _item_profiles_indices, _item_profiles_values, _item_profiles_weights, _item_profiles_shape = iterator.get_next()
        self.initializer = iterator.initializer
        self.labels = _labels
        self.userIds = _userIds
        self.itemIds = _itemIds
        self.user_profiles_indices = _user_profiles_indices
        self.user_profiles_values = _user_profiles_values
        self.user_profiles_weights = _user_profiles_weights
        self.user_profiles_shape = _user_profiles_shape
        self.item_profiles_indices = _item_profiles_indices
        self.item_profiles_values = _item_profiles_values
        self.item_profiles_weights = _item_profiles_weights
        self.item_profiles_shape = _item_profiles_shape

    def parser(self, record):
        keys_to_features = {
            'labels': tf.FixedLenFeature([], tf.string),
            'userIds': tf.VarLenFeature(tf.int64),
            'itemIds': tf.VarLenFeature(tf.int64),
            'user_profiles_indices': tf.FixedLenFeature([], tf.string),
            'user_profiles_values': tf.VarLenFeature(tf.int64),
            'user_profiles_weights': tf.VarLenFeature(tf.float32),
            'user_profiles_shape': tf.FixedLenFeature([2], tf.int64),
            'item_profiles_indices': tf.FixedLenFeature([], tf.string),
            'item_profiles_values': tf.VarLenFeature(tf.int64),
            'item_profiles_weights': tf.VarLenFeature(tf.float32),
            'item_profiles_shape': tf.FixedLenFeature([2], tf.int64)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        labels = tf.reshape(tf.decode_raw(parsed['labels'], tf.float32), [-1, 1])
        userIds = tf.sparse_tensor_to_dense(parsed['userIds'])
        itemIds = tf.sparse_tensor_to_dense(parsed['itemIds'])

        user_profiles_indices = tf.reshape(tf.decode_raw(parsed['user_profiles_indices'], tf.int64), [-1, 2])
        user_profiles_values = tf.sparse_tensor_to_dense(parsed['user_profiles_values'])
        user_profiles_weights = tf.sparse_tensor_to_dense(parsed['user_profiles_weights'])
        user_profiles_shape = parsed['user_profiles_shape']

        item_profiles_indices = tf.reshape(tf.decode_raw(parsed['item_profiles_indices'], tf.int64), [-1, 2])
        item_profiles_values = tf.sparse_tensor_to_dense(parsed['item_profiles_values'])
        item_profiles_weights = tf.sparse_tensor_to_dense(parsed['item_profiles_weights'])
        item_profiles_shape = parsed['item_profiles_shape']

        return labels, userIds, itemIds, \
               user_profiles_indices, user_profiles_values, user_profiles_weights, user_profiles_shape, \
               item_profiles_indices, item_profiles_values, item_profiles_weights, item_profiles_shape

class DKNIterator(BaseIterator):
    def __init__(self, src_dataset, hparams):
        self.batch_size = hparams.batch_size
        self.doc_size = hparams.doc_size
        self.get_iterator(src_dataset)

    def get_iterator(self, src_dataset):
        src_dataset = src_dataset.map(self.parser)
        # shuffle
        # src_dataset = src_dataset.shuffle(buffer_size=BUFFER_SIZE)
        iterator = src_dataset.make_initializable_iterator()
        _candidate_news_index_batch, _candidate_news_val_batch, _labels, \
        _click_news_indices, _click_news_values, \
        _click_news_weights, _click_news_shape,\
        _candidate_news_entity_index_batch, _click_news_entity_values\
            = iterator.get_next()

        self.initializer = iterator.initializer
        self.candidate_news_index_batch = _candidate_news_index_batch
        self.candidate_news_val_batch = _candidate_news_val_batch
        self.labels = _labels
        self.click_news_indices = _click_news_indices
        self.click_news_values = _click_news_values
        self.click_news_weights = _click_news_weights
        self.click_news_shape = _click_news_shape
        # === begin ===
        self.candidate_news_entity_index_batch = _candidate_news_entity_index_batch
        self.click_news_entity_values = _click_news_entity_values
        # === end ===

    def parser(self, record):
        keys_to_features = {
            'candidate_news_index_batch': tf.FixedLenFeature([], tf.string),
            'candidate_news_val_batch': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),
            'click_news_indices': tf.FixedLenFeature([], tf.string),
            'click_news_values': tf.VarLenFeature(tf.int64),
            'click_news_weights': tf.VarLenFeature(tf.float32),
            'click_news_shape': tf.FixedLenFeature([2], tf.int64),
            # === begin ===
            'candidate_news_entity_index_batch': tf.FixedLenFeature([], tf.string),
            'click_news_entity_values': tf.VarLenFeature(tf.int64)
            # === end ===
        }

        parsed = tf.parse_single_example(record, keys_to_features)
        candidate_news_index_batch = tf.reshape(tf.decode_raw(parsed['candidate_news_index_batch'], tf.int64),
                                                [self.batch_size, self.doc_size])
        candidate_news_val_batch = tf.reshape(tf.decode_raw(parsed['candidate_news_val_batch'], tf.float32),
                                              [self.batch_size, self.doc_size])
        labels = tf.reshape(tf.decode_raw(parsed['labels'], tf.float32), [-1, 1])
        click_news_indices = tf.reshape(tf.decode_raw(parsed['click_news_indices'], tf.int64), [-1, 2])
        click_news_values = tf.sparse_tensor_to_dense(parsed['click_news_values'])
        click_news_weights = tf.sparse_tensor_to_dense(parsed['click_news_weights'])
        click_news_shape = parsed['click_news_shape']
        # === begin ===
        candidate_news_entity_index_batch = tf.reshape(tf.decode_raw(parsed['candidate_news_entity_index_batch'], tf.int64),
                                                [-1, self.doc_size])
        click_news_entity_values = tf.sparse_tensor_to_dense(parsed['click_news_entity_values'])
        # === end ===

        return candidate_news_index_batch, candidate_news_val_batch, labels, \
               click_news_indices, click_news_values, \
               click_news_weights, click_news_shape, \
               candidate_news_entity_index_batch, click_news_entity_values

class MKRIterator(BaseIterator):
    def __init__(self, src_dataset_rs, src_dataset_kg):
        self.get_iterator_rs(src_dataset_rs)
        self.get_iterator_kg(src_dataset_kg)
        self.initializer = [self.initializer_kg, self.initializer_rs]


    def get_iterator_rs(self, src_dataset):
        src_dataset = src_dataset.map(self.parser_rs)
        # shuffle
        # src_dataset = src_dataset.shuffle(buffer_size=BUFFER_SIZE)
        iterator = src_dataset.make_initializable_iterator()
        ratings, user_ids, item_ids = iterator.get_next()

        self.initializer_rs = iterator.initializer
        self.labels = ratings
        self.users = user_ids
        self.items = item_ids

    def parser_rs(self, record):
        keys_to_features = {
            'ratings': tf.FixedLenFeature([], tf.string),
            'user_ids': tf.FixedLenFeature([], tf.string),
            'item_ids': tf.FixedLenFeature([], tf.string),
        }

        parsed = tf.parse_single_example(record, keys_to_features)
        ratings = tf.reshape(tf.decode_raw(parsed['ratings'], tf.float32), [-1])
        user_ids = tf.reshape(tf.decode_raw(parsed['user_ids'], tf.int32), [-1])
        item_ids = tf.reshape(tf.decode_raw(parsed['item_ids'], tf.int32), [-1])

        return ratings, user_ids, item_ids

    def get_iterator_kg(self, src_dataset):
        src_dataset = src_dataset.map(self.parser_kg)
        # shuffle
        # src_dataset = src_dataset.shuffle(buffer_size=BUFFER_SIZE)
        iterator = src_dataset.make_initializable_iterator()
        relations, heads, tails = iterator.get_next()

        self.initializer_kg = iterator.initializer
        self.relations = relations
        self.heads = heads
        self.tails = tails

    def parser_kg(self, record):
        keys_to_features = {
            'relations': tf.FixedLenFeature([], tf.string),
            'heads': tf.FixedLenFeature([], tf.string),
            'tails': tf.FixedLenFeature([], tf.string),
        }

        parsed = tf.parse_single_example(record, keys_to_features)
        relations = tf.reshape(tf.decode_raw(parsed['relations'], tf.int32), [-1])
        heads = tf.reshape(tf.decode_raw(parsed['heads'], tf.int32), [-1])
        tails = tf.reshape(tf.decode_raw(parsed['tails'], tf.int32), [-1])

        return relations, heads, tails


class RippleIterator(BaseIterator):
    def __init__(self, src_dataset, hparams):
        self.batch_size = hparams.batch_size
        self.hparams = hparams
        self.get_iterator(src_dataset)

    def get_iterator(self, src_dataset):
        src_dataset = src_dataset.map(self.parser)
        # shuffle
        # src_dataset = src_dataset.shuffle(buffer_size=BUFFER_SIZE)
        iterator = src_dataset.make_initializable_iterator()
        item, label, mem_h, mem_r, mem_t, mem_len = iterator.get_next()

        self.initializer = iterator.initializer
        self.item = item
        self.labels = label
        self.mem_h = mem_h
        self.mem_r = mem_r
        self.mem_t = mem_t
        # self.mem_mask = mem_mask
        self.mem_len = mem_len

    def parser(self, record):
        keys_to_features = {
            'item': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        }
        for hop in range(self.hparams.n_hops):
            keys_to_features["memory_head_" + str(hop)] = tf.FixedLenFeature([], tf.string)
            keys_to_features["memory_relation_" + str(hop)] = tf.FixedLenFeature([], tf.string)
            keys_to_features["memory_tail_" + str(hop)] = tf.FixedLenFeature([], tf.string)
            # keys_to_features["memory_mask_" + str(hop)] = tf.FixedLenFeature([], tf.string)
            keys_to_features["memory_length_" + str(hop)] = tf.FixedLenFeature([], tf.string)


        parsed = tf.parse_single_example(record, keys_to_features)
        item = tf.reshape(tf.decode_raw(parsed["item"], tf.int32), [-1])
        label = tf.reshape(tf.decode_raw(parsed["label"], tf.float32), [-1])
        mem_h = []
        mem_t = []
        mem_r = []
        mem_len = []
        mem_mask = []
        for hop in range(self.hparams.n_hops):
            mem_h.append(
                tf.reshape(tf.decode_raw(parsed["memory_head_" + str(hop)], tf.int32), [-1, self.hparams.n_memory]))
            mem_r.append(
                tf.reshape(tf.decode_raw(parsed["memory_relation_" + str(hop)], tf.int32), [-1, self.hparams.n_memory]))
            mem_t.append(
                tf.reshape(tf.decode_raw(parsed["memory_tail_" + str(hop)], tf.int32), [-1, self.hparams.n_memory]))
            # mem_mask.append(
            #     tf.reshape(tf.decode_raw(parsed["memory_mask_" + str(hop)], tf.float32), [-1, self.hparams.n_memory]))
            mem_len.append(
                tf.reshape(tf.decode_raw(parsed["memory_length_" + str(hop)], tf.int32), [-1]))
        mem_h = tf.stack(mem_h)
        mem_r = tf.stack(mem_r)
        mem_t = tf.stack(mem_t)
        mem_len = tf.stack(mem_len)
        return item, label, mem_h, mem_r, mem_t, mem_len

