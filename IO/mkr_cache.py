from IO.base_cache import BaseCache
import tensorflow as tf
import numpy as np
from collections import defaultdict
import utils.util as util

__all__ = ["MKRCache"]


class MKRCache(BaseCache):
    def _load_rating_data_from_batch(self, hparams, file):
        batch_size = hparams.batch_size
        ratings = []
        user_ids = []
        item_ids = []
        impression_ids = []
        cnt = 0
        with open(file, "r", encoding="utf8") as rfile:
            for line in rfile:
                line = line.strip()
                tmp = line.strip().split(util.USER_ID_SPLIT)
                if len(tmp) == 2:
                    impression_ids.append(tmp[1].strip())

                line = tmp[0]
                line = line.split("\t")
                if not line:
                    break
                user_id = int(line[0])
                item_id = int(line[1])
                rating = float(line[2])
                ratings.append(rating)
                user_ids.append(user_id)
                item_ids.append(item_id)
                cnt += 1
                if cnt == batch_size:
                    yield user_ids, item_ids, ratings, impression_ids
                    ratings = []
                    user_ids = []
                    item_ids = []
                    impression_ids = []
                    cnt = 0
            if cnt > 0:
                yield user_ids, item_ids, ratings, impression_ids

    def _load_kg_data_from_batch(self, hparams, file):
        batch_size = hparams.batch_size
        relations = []
        heads = []
        tails =[]
        cnt = 0
        with open(file, "r", encoding="utf8") as rfile:
            for line in rfile:
                line = line.strip().split("\t")
                if not line:
                    break
                head = int(line[0])
                tail = int(line[1])
                relation = int(line[2])
                heads.append(head)
                tails.append(tail)
                relations.append(relation)
                cnt += 1
                if cnt == batch_size:
                    yield heads, tails, relations
                    relations = []
                    heads = []
                    tails = []
                    cnt = 0
            if cnt > 0:
                yield heads, tails, relations

    def write_tfrecord(self, infile, outfile, hparams):
        sample_num = 0
        impression_id_list = []
        writer = tf.python_io.TFRecordWriter(outfile)
        try:
            for user_ids, item_ids, ratings, impression_ids in self._load_rating_data_from_batch(hparams, infile):
                sample_num += len(ratings)
                impression_id_list.extend(impression_ids)
                user_ids = np.asarray(user_ids, dtype=np.int32)
                user_str = user_ids.tostring()
                item_ids = np.asarray(item_ids, dtype=np.int32)
                item_str = item_ids.tostring()
                ratings = np.asarray(ratings, dtype=np.float32)
                rating_str = ratings.tostring()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'user_ids': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[user_str])),
                            'item_ids': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[item_str])),
                            'ratings': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[rating_str]))
                        }
                    )
                )
                serialized = example.SerializeToString()
                writer.write(serialized)
        except:
            raise ValueError('train data format must be mkr, for example: user\titem\trating')
        writer.close()
        return sample_num, impression_id_list

    def write_kg_tfrecord(self, infile, outfile, hparams):
        writer = tf.python_io.TFRecordWriter(outfile)
        try:
            for heads, tails, relations in self._load_kg_data_from_batch(hparams, infile):
                heads = np.asarray(heads, dtype=np.int32)
                head_str = heads.tostring()
                tails = np.asarray(tails, dtype=np.int32)
                tail_str = tails.tostring()
                relations = np.asarray(relations, dtype=np.int32)
                relation_str = relations.tostring()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'heads': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[head_str])),
                            'tails': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tail_str])),
                            'relations': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[relation_str]))
                        }
                    )
                )
                serialized = example.SerializeToString()
                writer.write(serialized)
        except:
            raise ValueError('train data format must be mkr, for example: heads\ttails\trelations')
        writer.close()

    def _load_stat(self):
        # read rating file
        ratings_df = pd.read_table(config.ratings_file, header=None, names=['user_id', 'item_id', 'rating'])
        n_ratings = ratings_df.shape[0]
        ratings = np.concatenate((np.array(ratings_df['user_id']).reshape(n_ratings, 1),
                                  np.array(ratings_df['item_id']).reshape(n_ratings, 1),
                                  np.array(ratings_df['rating']).reshape(n_ratings, 1)),
                                 axis=1)

        n_users = np.unique(ratings[:, 0]).shape[0]
        n_items = np.unique(ratings[:, 1]).shape[0]

        # read KG file
        kg_df = pd.read_table(config.kg_file, header=None, names=['head', 'tail', 'relation'])
        n_kg = kg_df.shape[0]
        kg = np.concatenate((np.array(kg_df['head']).reshape(n_kg, 1),
                             np.array(kg_df['tail']).reshape(n_kg, 1),
                             np.array(kg_df['relation']).reshape(n_kg, 1)),
                            axis=1)
        n_entities = np.unique(np.concatenate((ratings[:, 1], kg[:, 0], kg[:, 1]))).shape[0]
        n_relations = np.unique(kg[:, 2]).shape[0]

        return n_users, n_items, n_entities, n_relations