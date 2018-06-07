"""define dkn cache class for reading data"""
from IO.base_cache import BaseCache
import tensorflow as tf
import numpy as np
import random
import utils.util as util

__all__ = ["DKNCache"]


# word index start by 1
# 每个document的词数是一致的,次数可以设为L = 5
class DKNCache(BaseCache):
    @staticmethod
    def _load_batch_data_from_file(file, hparams):
        batch_size = hparams.batch_size
        labels = []
        candidate_news_index_batch = []
        candidate_news_val_batch = []
        click_news_index_batch = []
        click_news_val_batch = []
        # === begin ===
        candidate_news_entity_index_batch = []
        click_news_entity_index_batch = []
        impression_id = []
        # === end ===
        cnt = 0
        with open(file, 'r') as reader:
            for line in reader:
                tmp = line.strip().split(util.USER_ID_SPLIT)
                if len(tmp) == 2:
                    impression_id.append(tmp[1].strip())
                cnt += 1
                arr = line.split(' ')
                label = float(arr[0])
                candidate_news_index = []
                candidate_news_val = []
                click_news_index = []
                click_news_val = []
                # === begin ===
                candidate_news_entity_index = []
                click_news_entity_index = []
                # === end ===
                for news in arr[1:]:
                    tokens = news.split(':')
                    if tokens[0] == 'CandidateNews':
                        # word index start by 0
                        for item in tokens[1].split(','):
                            candidate_news_index.append(int(item))
                            candidate_news_val.append(float(1))
                    elif 'clickedNews' in tokens[0]:
                        for item in tokens[1].split(','):
                            click_news_index.append(int(item))
                            click_news_val.append(float(1))
                    # === begin ===
                    elif tokens[0] == 'entity':
                        for item in tokens[1].split(','):
                            candidate_news_entity_index.append(int(item))
                    elif 'entity' in tokens[0]:
                        for item in tokens[1].split(','):
                            click_news_entity_index.append(int(item))
                    # === end ===
                    else:
                        raise ValueError("data format is wrong")

                candidate_news_index_batch.append(candidate_news_index)
                candidate_news_val_batch.append(candidate_news_val)
                click_news_index_batch.append(click_news_index)
                click_news_val_batch.append(click_news_val)
                # === begin ===
                candidate_news_entity_index_batch.append(candidate_news_entity_index)
                click_news_entity_index_batch.append(click_news_entity_index)
                # === end ===
                labels.append(label)

                if cnt == batch_size:
                    # === begin ===
                    yield labels,\
                          candidate_news_index_batch, candidate_news_val_batch, \
                          click_news_index_batch, click_news_val_batch, \
                          candidate_news_entity_index_batch, \
                          click_news_entity_index_batch, impression_id
                    # === end ===
                    labels = []
                    candidate_news_index_batch = []
                    candidate_news_val_batch = []
                    click_news_index_batch = []
                    click_news_val_batch = []
                    cnt = 0
                    # === begin ===
                    candidate_news_entity_index_batch = []
                    click_news_entity_index_batch = []
                    # === end ===

    @staticmethod
    def _convert_data(labels,
                      candidate_news_index_batch, candidate_news_val_batch,
                      click_news_index_batch, click_news_val_batch,
                      candidate_news_entity_index_batch,
                      click_news_entity_index_batch,
                      hparams):
        batch_size = hparams.batch_size
        instance_cnt = len(labels)

        click_news_indices = []
        click_news_values = []
        click_news_weights = []
        click_news_shape = [batch_size, -1]
        # === begin ===
        click_news_entity_values = []
        # === end ===

        #处理click_field_shape的数据,构造稀疏的矩阵来进行存储
        batch_max_len = 0
        for i in range(instance_cnt):
            m = len(click_news_index_batch[i])
            batch_max_len = m if m > batch_max_len else batch_max_len
            for j in range(m):
                click_news_indices.append([i, j])
                click_news_values.append(click_news_index_batch[i][j])
                click_news_weights.append(click_news_val_batch[i][j])
                # === begin ===
                click_news_entity_values.append(
                    click_news_entity_index_batch[i][j])
                # === end ===
        click_news_shape[1] = batch_max_len

        res = {}
        res['labels'] = np.asarray([[label] for label in labels], dtype=np.float32)
        res['candidate_news_index_batch'] = np.asarray(candidate_news_index_batch, dtype=np.int64)
        res['candidate_news_val_batch'] = np.asarray(candidate_news_val_batch, dtype=np.float32)
        res['click_news_indices'] = np.asarray(click_news_indices, dtype=np.int64)
        res['click_news_values'] = np.asarray(click_news_values, dtype=np.int64)
        res['click_news_weights'] = np.asarray(click_news_weights, dtype=np.float32)
        res['click_news_shape'] = np.asarray(click_news_shape, dtype=np.int64)
        # === begin ===
        res['candidate_news_entity_index_batch'] = np.asarray(
            candidate_news_entity_index_batch, dtype=np.int64)
        res['click_news_entity_values'] = np.asarray(
            click_news_entity_values, dtype=np.int64)
        # === end ===
        return res

    def write_tfrecord(self, infile, outfile, hparams):
        writer = tf.python_io.TFRecordWriter(outfile)
        sample_num = 0
        impression_id_list = []
        for index, (labels,
                    candidate_news_index_batch, candidate_news_val_batch,
                    click_news_index_batch, click_news_val_batch,
                    # === begin ===
                    candidate_news_entity_index_batch,
                    click_news_entity_index_batch,
                    impression_id
                    # === end ===
                    ) \
                    in enumerate(self._load_batch_data_from_file(infile, hparams)):
            sample_num += len(labels)
            impression_id_list.extend(impression_id)
            input_in_sp = self._convert_data(
                labels,
                candidate_news_index_batch, candidate_news_val_batch,
                click_news_index_batch, click_news_val_batch,
                # === begin ===
                candidate_news_entity_index_batch,
                click_news_entity_index_batch,
                # === end ===
                hparams)

            labels = input_in_sp['labels']
            candidate_news_index_batch = input_in_sp['candidate_news_index_batch']
            candidate_news_val_batch = input_in_sp['candidate_news_val_batch']

            click_news_indices = input_in_sp['click_news_indices']
            click_news_values = input_in_sp['click_news_values']
            click_news_weights = input_in_sp['click_news_weights']
            click_news_shape = input_in_sp['click_news_shape']

            # === begin ===
            candidate_news_entity_index_batch =\
                input_in_sp['candidate_news_entity_index_batch']
            click_news_entity_values = input_in_sp['click_news_entity_values']
            # === end ===


            labels_str = labels.tostring()
            candidate_news_index_batch_str = candidate_news_index_batch.tostring()
            candidate_news_val_batch_str = candidate_news_val_batch.tostring()
            click_news_indices_str = click_news_indices.tostring()

            # === begin ===
            candidate_news_entity_index_batch_str =\
                candidate_news_entity_index_batch.tostring()
            # === end ===


            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'candidate_news_index_batch': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[candidate_news_index_batch_str])),
                        'candidate_news_val_batch': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[candidate_news_val_batch_str])),
                        'labels': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[labels_str])),
                        'click_news_indices': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[click_news_indices_str])),
                        'click_news_values': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=click_news_values)),
                        'click_news_weights': tf.train.Feature(
                            float_list=tf.train.FloatList(value=click_news_weights)),
                        'click_news_shape': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=click_news_shape)),
                        # === begin ===
                        'candidate_news_entity_index_batch': tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[candidate_news_entity_index_batch_str])),
                        'click_news_entity_values': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=click_news_entity_values)),
                        # === end ===
                    }
                )
            )
            serialized = example.SerializeToString()
            writer.write(serialized)
        return sample_num, impression_id_list


def test_cache():
    def create_hparams():
        """Create hparams."""
        return tf.contrib.training.HParams(
            batch_size=2
        )

    hparams = create_hparams()
    infile = '../data/toy'
    outfile = '../cache/toy.tfrecord'
    cache = DKNCache()
    cache.write_tfrecord(infile, outfile, hparams)

# test_cache()
