"""define FfmCache class for cache the format dataset"""
from IO.base_cache import BaseCache
import tensorflow as tf
import numpy as np
import utils.util as util

__all__ = ["CCCFNetCache"]


class CCCFNetCache(BaseCache):
    # feat index start by 1
    def _load_batch_data_from_file(self, file, hparams):
        batch_size = hparams.batch_size
        labels = []
        userIds = []
        itemIds = []
        user_profiles = []
        item_profiles = []
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
                cols = line.strip().split('|')

                label = float(cols[0].strip())
                userId = int(cols[1].strip())
                itemId = int(cols[2].strip())

                cur_user_profile = []
                user_feat = cols[3].strip(' ')
                if not user_feat:
                    cur_user_profile.append([0, 0.0])
                else:
                    for items in user_feat.split(','):
                        tokens = items.strip().split(':')
                        cur_user_profile.append([int(tokens[0]), float(tokens[1])])

                cur_item_profile = []
                item_feat = cols[4].strip(' ')
                if not item_feat:
                    cur_item_profile.append([0, 0.0])
                else:
                    for items in item_feat.split(','):
                        tokens = items.strip().split(':')
                        cur_item_profile.append([int(tokens[0]), float(tokens[1])])

                labels.append(label)
                userIds.append(userId)
                itemIds.append(itemId)
                user_profiles.append(cur_user_profile)
                item_profiles.append(cur_item_profile)
                cnt += 1
                if cnt == batch_size:
                    yield labels, userIds, itemIds, user_profiles, item_profiles, impression_id
                    labels = []
                    userIds = []
                    itemIds = []
                    user_profiles = []
                    item_profiles = []
                    impression_id = []
                    cnt = 0
        if cnt > 0:
            yield labels, userIds, itemIds, user_profiles, item_profiles, impression_id

    def _convert_data(self, labels, userIds, itemIds, user_profiles, item_profiles, hparams):
        instance_cnt = len(labels)
        user_profiles_indices = []
        user_profiles_values = []
        user_profiles_weights = []
        user_profiles_shape = [instance_cnt, -1]
        for i in range(instance_cnt):
            m = len(user_profiles[i])
            for j in range(m):
                user_profiles_indices.append([i, j])
                user_profiles_values.append(user_profiles[i][j][0])
                user_profiles_weights.append(user_profiles[i][j][1])
            user_profiles_shape[1] = max(user_profiles_shape[1], m)

        item_profiles_indices = []
        item_profiles_values = []
        item_profiles_weights = []
        item_profiles_shape = [instance_cnt, -1]
        for i in range(instance_cnt):
            m = len(item_profiles[i])
            for j in range(m):
                item_profiles_indices.append([i, j])
                item_profiles_values.append(item_profiles[i][j][0])
                item_profiles_weights.append(item_profiles[i][j][1])
            item_profiles_shape[1] = max(item_profiles_shape[1], m)

        res = {}
        res['labels'] = np.asarray([[label] for label in labels], dtype=np.float32)
        res['userIds'] = np.asarray(userIds, dtype=np.int64)
        res['itemIds'] = np.asarray(itemIds, dtype=np.int64)

        sorted_index = sorted(range(len(user_profiles_indices)),
                              key=lambda k: (user_profiles_indices[k][0], \
                                             user_profiles_indices[k][1]))
        res['user_profiles_indices'] = np.asarray(user_profiles_indices, dtype=np.int64)[sorted_index]
        res['user_profiles_values'] = np.asarray(user_profiles_values, dtype=np.int64)[sorted_index]
        res['user_profiles_weights'] = np.asarray(user_profiles_weights, dtype=np.float32)[sorted_index]
        res['user_profiles_shape'] = np.asarray(user_profiles_shape, dtype=np.int64)

        sorted_index = sorted(range(len(item_profiles_indices)),
                              key=lambda k: (item_profiles_indices[k][0], \
                                             item_profiles_indices[k][1]))
        res['item_profiles_indices'] = np.asarray(item_profiles_indices, dtype=np.int64)[sorted_index]
        res['item_profiles_values'] = np.asarray(item_profiles_values, dtype=np.int64)[sorted_index]
        res['item_profiles_weights'] = np.asarray(item_profiles_weights, dtype=np.float32)[sorted_index]
        res['item_profiles_shape'] = np.asarray(item_profiles_shape, dtype=np.int64)
        return res

    def write_tfrecord(self, infile, outfile, hparams):
        sample_num = 0
        impression_id_list = []
        writer = tf.python_io.TFRecordWriter(outfile)
        try:
            for (labels, userIds, itemIds, user_profiles, item_profiles,
                 impression_id) in self._load_batch_data_from_file(
                infile, hparams):
                impression_id_list.extend(impression_id)
                input_in_sp = self._convert_data(labels, userIds, itemIds, user_profiles, item_profiles, hparams)
                labels = input_in_sp['labels']
                userIds = input_in_sp['userIds']
                itemIds = input_in_sp['itemIds']

                user_profiles_indices = input_in_sp['user_profiles_indices']
                user_profiles_values = input_in_sp['user_profiles_values']
                user_profiles_weights = input_in_sp['user_profiles_weights']
                user_profiles_shape = input_in_sp['user_profiles_shape']

                item_profiles_indices = input_in_sp['item_profiles_indices']
                item_profiles_values = input_in_sp['item_profiles_values']
                item_profiles_weights = input_in_sp['item_profiles_weights']
                item_profiles_shape = input_in_sp['item_profiles_shape']

                labels_str = labels.tostring()

                user_profiles_indices_str = user_profiles_indices.tostring()
                item_profiles_indices_str = item_profiles_indices.tostring()

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'labels': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[labels_str])),
                            'userIds': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=userIds)),
                            'itemIds': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=itemIds)),
                            'user_profiles_indices': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[user_profiles_indices_str])),
                            'user_profiles_values': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=user_profiles_values)),
                            'user_profiles_weights': tf.train.Feature(
                                float_list=tf.train.FloatList(value=user_profiles_weights)),
                            'user_profiles_shape': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=user_profiles_shape)),
                            'item_profiles_indices': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[item_profiles_indices_str])),
                            'item_profiles_values': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=item_profiles_values)),
                            'item_profiles_weights': tf.train.Feature(
                                float_list=tf.train.FloatList(value=item_profiles_weights)),
                            'item_profiles_shape': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=item_profiles_shape))
                        }
                    )
                )
                serialized = example.SerializeToString()
                writer.write(serialized)
                sample_num += len(labels)
        except:
            raise ValueError('train data format must be cccfnet format')
        return sample_num, impression_id_list
