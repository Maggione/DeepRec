"""define deep interest network cache class for reading data"""
from IO.base_cache import BaseCache
import tensorflow as tf
import numpy as np
from collections import defaultdict
import utils.util as util

__all__ = ["DinCache"]


class DinCache(BaseCache):
    def _load_batch_data_from_file(self, file, hparams):
        batch_size = hparams.batch_size
        labels = []
        attention_news_features = []
        attention_user_features = []
        fm_features = []
        dnn_features = []
        impression_id = []
        cnt = 0
        with open(file, 'r') as rd:
            for line in rd:
                line = line.strip(' ')
                if not line:
                    break
                tmp = line.strip().split(util.USER_ID_SPLIT)
                if len(tmp) == 2:
                    impression_id.append(tmp[1].strip())
                line = tmp[0]
                cnt += 1
                cols = line.strip().split(' ')
                label = float(cols[0])
                if label > 0:
                    label = 1
                else:
                    label = 0
                news_feature_list = []
                attention_user_features_list = []
                fm_feature_list = []
                dnn_feature_list = []
                for word in cols[1:]:
                    if not word:
                        continue
                    tokens = word.split(':')
                    if tokens[0].strip().split(util.DIN_FORMAT_SPLIT)[0] == 'News':
                        news_feature_list.append([int(tokens[0].strip().split(util.DIN_FORMAT_SPLIT)[-1]) - 1, \
                                                  int(tokens[1]) - 1, \
                                                  float(tokens[2])])

                    if tokens[0].strip().split(util.DIN_FORMAT_SPLIT)[0] == 'User':
                        attention_user_features_list.append([ \
                            int(tokens[0].strip().split(util.DIN_FORMAT_SPLIT)[-1]) - 1, \
                            int(tokens[1]) - 1, \
                            float(tokens[2])])

                    fm_feature_list.append([int(tokens[1]) - 1, float(tokens[2])])

                    if ('News' not in tokens[0]) and ('User' not in tokens[0]):
                        dnn_feature_list.append([int(tokens[0]) - 1, \
                                                 int(tokens[1]) - 1, \
                                                 float(tokens[2])])

                attention_news_features.append(news_feature_list)
                attention_user_features.append(attention_user_features_list)
                fm_features.append(fm_feature_list)
                dnn_features.append(dnn_feature_list)
                labels.append(label)
                if cnt == batch_size:
                    yield labels, attention_news_features, attention_user_features, \
                          fm_features, dnn_features, impression_id
                    labels = []
                    attention_news_features = []
                    attention_user_features = []
                    fm_features = []
                    dnn_features = []
                    impression_id = []
                    cnt = 0
        if cnt > 0:
            yield labels, attention_news_features, attention_user_features, \
                  fm_features, dnn_features, impression_id
            labels = []
            attention_news_features = []
            attention_user_features = []
            fm_features = []
            dnn_features = []
            impression_id = []
            cnt = 0

    def _convert_data(self, labels, attention_news_features, attention_user_features, \
                      fm_features, dnn_features, hparams):
        dim = hparams.FEATURE_COUNT
        PAIR_NUM = hparams.PAIR_NUM
        DNN_FIELD_NUM = hparams.DNN_FIELD_NUM

        instance_cnt = len(labels)

        attention_news_indices = []
        attention_news_values = []
        attention_news_shape = [instance_cnt * PAIR_NUM, dim]

        attention_user_indices = []
        attention_user_values = []
        attention_user_weights = []
        attention_user_shape = [instance_cnt * PAIR_NUM, -1]

        fm_feat_indices = []
        fm_feat_val = []
        fm_feat_shape = [instance_cnt, dim]

        dnn_feat_indices = []
        dnn_feat_values = []
        dnn_feat_weight = []
        dnn_feat_shape = [instance_cnt * DNN_FIELD_NUM, -1]

        # PAIR_NUM is the parameter for attention field
        for i in range(instance_cnt):
            m = len(attention_news_features[i])
            for j in range(m):
                attention_news_indices.append(
                    [i * PAIR_NUM + attention_news_features[i][j][0], \
                     attention_news_features[i][j][1]])
                attention_news_values.append(attention_news_features[i][j][2])

        for i in range(instance_cnt):
            m = len(attention_user_features[i])
            field2feature_dict = {}
            for j in range(m):
                if attention_user_features[i][j][0] not in field2feature_dict:
                    field2feature_dict[attention_user_features[i][j][0]] = 0
                else:
                    field2feature_dict[attention_user_features[i][j][0]] += 1
                attention_user_indices.append(
                    [i * PAIR_NUM + attention_user_features[i][j][0],
                     field2feature_dict[attention_user_features[i][j][0]]])
                attention_user_values.append(attention_user_features[i][j][1])
                attention_user_weights.append(attention_user_features[i][j][2])
                if attention_user_shape[1] < field2feature_dict[attention_user_features[i][j][0]]:
                    attention_user_shape[1] = field2feature_dict[attention_user_features[i][j][0]]
        attention_user_shape[1] += 1

        # 处理FM的数据：fm_features[i][k]=[feat_index, feat_value]
        for i in range(instance_cnt):
            # 对FM的feature的组织形式
            n = len(fm_features[i])
            for k in range(n):
                fm_feat_indices.append([i, fm_features[i][k][0]])
                fm_feat_val.append(fm_features[i][k][1])

        # 处理非attention单元的field，需要设置参数DNN_FIELD_NUM，关键参数
        for i in range(instance_cnt):
            m = len(dnn_features[i])
            field2feature_dict = {}
            for j in range(m):
                if dnn_features[i][j][0] not in field2feature_dict:
                    field2feature_dict[dnn_features[i][j][0]] = 0
                else:
                    field2feature_dict[dnn_features[i][j][0]] += 1
                dnn_feat_indices.append(
                    [i * DNN_FIELD_NUM + dnn_features[i][j][0], \
                     field2feature_dict[dnn_features[i][j][0]]])
                dnn_feat_values.append(dnn_features[i][j][1])
                dnn_feat_weight.append(dnn_features[i][j][2])
                if dnn_feat_shape[1] < field2feature_dict[dnn_features[i][j][0]]:
                    dnn_feat_shape[1] = field2feature_dict[dnn_features[i][j][0]]
        dnn_feat_shape[1] += 1

        res = {}
        sorted_attention_news_indices = \
            sorted(range(len(attention_news_indices)), \
                   key=lambda k: (attention_news_indices[k][0], attention_news_indices[k][1]))
        res['attention_news_indices'] = np.asarray(attention_news_indices, dtype=np.int64)[ \
            sorted_attention_news_indices]
        res['attention_news_values'] = np.asarray(attention_news_values, dtype=np.float32)[ \
            sorted_attention_news_indices]
        res['attention_news_shape'] = np.asarray(attention_news_shape, dtype=np.int64)

        res['labels'] = np.asarray([[label] for label in labels], dtype=np.float32)

        sorted_attention_user_index = \
            sorted(range(len(attention_user_indices)),
                   key=lambda k: (attention_user_indices[k][0], attention_user_indices[k][1]))
        res['attention_user_indices'] = np.asarray(attention_user_indices, \
                                                   dtype=np.int64)[sorted_attention_user_index]
        res['attention_user_values'] = np.asarray(attention_user_values, \
                                                  dtype=np.int64)[sorted_attention_user_index]
        res['attention_user_weights'] = np.asarray(attention_user_weights, \
                                                   dtype=np.float32)[sorted_attention_user_index]
        res['attention_user_shape'] = np.asarray(attention_user_shape, dtype=np.int64)

        res['fm_feat_indices'] = np.asarray(fm_feat_indices, dtype=np.int64)
        res['fm_feat_val'] = np.asarray(fm_feat_val, dtype=np.float32)
        res['fm_feat_shape'] = np.asarray(fm_feat_shape, dtype=np.int64)

        sorted_dnn_feat_indices = sorted(
            range(len(dnn_feat_indices)),
            key=lambda k: (dnn_feat_indices[k][0], dnn_feat_indices[k][1]))
        res['dnn_feat_indices'] = np.asarray(dnn_feat_indices, \
                                             dtype=np.int64)[sorted_dnn_feat_indices]
        res['dnn_feat_values'] = np.asarray(dnn_feat_values, \
                                            dtype=np.int64)[sorted_dnn_feat_indices]
        res['dnn_feat_weight'] = np.asarray(dnn_feat_weight, \
                                            dtype=np.float32)[sorted_dnn_feat_indices]
        res['dnn_feat_shape'] = np.asarray(dnn_feat_shape, dtype=np.int64)

        return res

    # 因为cache的机制是每次将一个batch_size的数据进行cache.使用了attention网络之后，batch_size是构建deep interest network的必要参数
    # 数据集最后一个batch的数据可能不够batch_size。因此，代码中用了一些trick进行补齐，主要是将第一个batch_size的数据存下来。
    def write_tfrecord(self, infile, outfile, hparams):
        writer = tf.python_io.TFRecordWriter(outfile)
        sample_num = 0
        impression_id_list = []
        for index, (labels, attention_news_features, attention_user_features, \
                    fm_features, dnn_features, impression_id) \
                in enumerate(self._load_batch_data_from_file(infile, hparams)):
            sample_num += len(labels)
            impression_id_list.extend(impression_id)
            if index == 0:
                if len(labels) < hparams.batch_size:
                    raise ValueError("data num is less than one batch_size {0}".format(len(labels)))
                cache_labels, cache_attention_news_features, \
                cache_attention_user_features, cache_fm_features, \
                cache_dnn_features = labels, attention_news_features, \
                                     attention_user_features, \
                                     fm_features, dnn_features

            # 使用cache的一个batch_size的数据进行补齐
            if len(labels) < hparams.batch_size:
                for i in range(len(labels)):
                    cache_labels[i] = labels[i]
                    cache_attention_news_features[i] = attention_news_features[i]
                    cache_attention_user_features[i] = attention_user_features[i]
                    cache_fm_features[i] = fm_features[i]
                    cache_dnn_features[i] = dnn_features[i]
                labels = cache_labels
                attention_news_features = cache_attention_news_features
                attention_user_features = cache_attention_user_features
                fm_features = cache_fm_features
                dnn_features = cache_dnn_features

            input_in_sp = self._convert_data(labels, attention_news_features, \
                                             attention_user_features, fm_features, dnn_features, hparams)

            attention_news_indices = input_in_sp['attention_news_indices']
            attention_news_values = input_in_sp['attention_news_values']
            attention_news_shape = input_in_sp['attention_news_shape']

            attention_user_indices = input_in_sp['attention_user_indices']
            attention_user_values = input_in_sp['attention_user_values']
            attention_user_weights = input_in_sp['attention_user_weights']
            attention_user_shape = input_in_sp['attention_user_shape']

            fm_feat_indices = input_in_sp['fm_feat_indices']
            fm_feat_val = input_in_sp['fm_feat_val']
            fm_feat_shape = input_in_sp['fm_feat_shape']

            labels = input_in_sp['labels']

            dnn_feat_indices = input_in_sp['dnn_feat_indices']
            dnn_feat_values = input_in_sp['dnn_feat_values']
            dnn_feat_weight = input_in_sp['dnn_feat_weight']
            dnn_feat_shape = input_in_sp['dnn_feat_shape']

            attention_news_indices_str = attention_news_indices.tostring()
            attention_user_indices_str = attention_user_indices.tostring()
            fm_feat_indices_str = fm_feat_indices.tostring()
            labels_str = labels.tostring()
            dnn_feat_indices_str = dnn_feat_indices.tostring()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'attention_news_indices': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[attention_news_indices_str])),
                        'attention_news_values': tf.train.Feature(
                            float_list=tf.train.FloatList(value=attention_news_values)),
                        'attention_news_shape': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=attention_news_shape)),

                        'attention_user_indices': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[attention_user_indices_str])),
                        'attention_user_values': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=attention_user_values)),
                        'attention_user_weights': tf.train.Feature(
                            float_list=tf.train.FloatList(value=attention_user_weights)),
                        'attention_user_shape': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=attention_user_shape)),

                        'fm_feat_indices': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[fm_feat_indices_str])),
                        'fm_feat_val': tf.train.Feature(
                            float_list=tf.train.FloatList(value=fm_feat_val)),
                        'fm_feat_shape': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=fm_feat_shape)),

                        'labels': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[labels_str])),

                        'dnn_feat_indices': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[dnn_feat_indices_str])),
                        'dnn_feat_values': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=dnn_feat_values)),
                        'dnn_feat_weight': tf.train.Feature(
                            float_list=tf.train.FloatList(value=dnn_feat_weight)),
                        'dnn_feat_shape': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=dnn_feat_shape))
                    }
                )
            )
            serialized = example.SerializeToString()
            writer.write(serialized)
        writer.close()
        return sample_num, impression_id_list
