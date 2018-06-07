from IO.base_cache import BaseCache
import tensorflow as tf
import numpy as np
from collections import defaultdict
import utils.util as util

__all__ = ["RippleCache"]


class RippleCache(BaseCache):
    def __init__(self, hparams):
        self.user_clicks = self._read_user_dict(hparams.user_clicks)  # dict
        self.KG = self._read_kg(hparams.kg_file)  # knowledge graphg
        self.user_click_limit = hparams.user_click_limit
        self.entity_limit = hparams.entity_limit
        self.kg_ratio = hparams.kg_ratio
        self.n_hops = hparams.n_hops
        self.n_memory = hparams.n_memory

    def _load_data_from_batch(self, file, hparams):
        """generate the next batch data

                :param interactions: list, element like [user, item, label]
                :param current_pos: int
                :returns  items:
                          labels:
                          initial_entity: list, length: batch_size
                """
        h_list = []
        t_list = []
        r_list = []
        items = []
        labels = []
        init_h = []
        init_t = []
        init_r = []
        impression_id = []
        cnt = 0
        with open(file, "r", encoding="utf8") as rfile:
            for line in rfile:
                line = line.strip()
                tmp = line.strip().split(util.USER_ID_SPLIT)
                if len(tmp) == 2:
                    impression_id.append(tmp[1].strip())

                line = tmp[0]
                line = line.split("\t")
                if not line:
                    break
                user = int(line[0])
                if user in self.user_clicks:
                    user_click = self.user_clicks[user]  # the items set of the user history
                else:
                    print("user not in the training dataset:" + str(user))
                    # assert 0 == 1
                    continue
                items.append(int(line[1]))
                labels.append(float(line[2]))

                degree = min(self.user_click_limit, len(user_click))
                user_click = user_click[:degree]
                init_h_tmp = []
                init_t_tmp = []
                init_r_tmp = []
                for click in user_click:  # item
                    degree = max(1, int(min(self.entity_limit, len(self.KG[click][0])) * self.kg_ratio))
                    init_h_tmp += degree * [click]
                    init_t_tmp += self.KG[click][0][0:degree]  # tail
                    init_r_tmp += self.KG[click][1][0:degree]  # relation
                init_h.append(init_h_tmp)
                init_t.append(init_t_tmp)
                init_r.append(init_r_tmp)
                cnt += 1
                if cnt == hparams.batch_size:
                    h_list.append(init_h)
                    t_list.append(init_t)
                    r_list.append(init_r)

                    for _ in range(self.n_hops - 1):
                        next_h, next_t, next_r = self.get_next_entity(t_list[-1])
                        h_list.append(next_h)
                        t_list.append(next_t)
                        r_list.append(next_r)

                    # get the memory length
                    mem_len_list = []
                    for i in range(self.n_hops):  # n_hops
                        mem_len_batch = []
                        for j in range(len(h_list[0])):  # batch_size
                            # padding
                            mem_len_batch.append(min(len(h_list[i][j]), self.n_memory))
                            h_list[i][j] = util.pad_seq(h_list[i][j], self.n_memory)
                            r_list[i][j] = util.pad_seq(r_list[i][j], self.n_memory)
                            t_list[i][j] = util.pad_seq(t_list[i][j], self.n_memory)
                        mem_len_list.append(mem_len_batch)
                    # mem_mask_list = []
                    # for i in range(self.n_hops):
                    #     mem_mask = np.zeros([len(items), self.n_memory])
                    #     for j in range(mem_mask.shape[0]):
                    #         mem_mask[j, mem_len_list[i][j]:] = -1e6
                    #     mem_mask_list.append(mem_mask)
                    yield items, labels, h_list, t_list, r_list, mem_len_list, impression_id# , mem_mask_list
                    h_list = []
                    t_list = []
                    r_list = []
                    items = []
                    labels = []
                    init_h = []
                    init_t = []
                    init_r = []
                    impression_id = []
                    cnt = 0
            if cnt > 0:
                h_list.append(init_h)
                t_list.append(init_t)
                r_list.append(init_r)

                for _ in range(self.n_hops - 1):
                    next_h, next_t, next_r = self.get_next_entity(t_list[-1])
                    h_list.append(next_h)
                    t_list.append(next_t)
                    r_list.append(next_r)

                # get the memory length
                mem_len_list = []
                for i in range(self.n_hops):  # n_hops
                    mem_len_batch = []
                    for j in range(len(h_list[0])):  # batch_size
                        # padding
                        mem_len_batch.append(min(len(h_list[i][j]), self.n_memory))
                        h_list[i][j] = util.pad_seq(h_list[i][j], self.n_memory)
                        r_list[i][j] = util.pad_seq(r_list[i][j], self.n_memory)
                        t_list[i][j] = util.pad_seq(t_list[i][j], self.n_memory)
                    mem_len_list.append(mem_len_batch)
                # mem_mask_list = []
                # for i in range(self.n_hops):
                #     mem_mask = np.zeros([len(items), self.n_memory])
                #     for j in range(mem_mask.shape[0]):
                #         mem_mask[j, mem_len_list[i][j]:] = -1e6
                #     mem_mask_list.append(mem_mask)
                yield items, labels, h_list, t_list, r_list, mem_len_list, impression_id # , mem_mask_list

    def get_next_entity(self, inner_entity):
        """get the next Mui

        From the movies that rated by the user, to extend.
        :param inner_entity: list, batch_size x n_memory, [t1, t2, t3, t4], batch_size x N
        :return next_entity:
                next_relation
        """

        next_h = []
        next_t = []
        next_r = []
        for user_entity in inner_entity:  # batch_size
            next_h_tmp = []
            next_t_tmp = []
            next_r_tmp = []
            for entity in user_entity:  #
                if entity in self.KG:
                    tmp_entity = self.KG[entity]
                    degree = min(self.entity_limit, len(tmp_entity[0]))
                    next_h_tmp += degree * [entity]
                    next_t_tmp += tmp_entity[0][0:degree]  # entities
                    next_r_tmp += tmp_entity[1][0:degree]  # relations

            next_h.append(next_h_tmp)
            next_t.append(next_t_tmp)
            next_r.append(next_r_tmp)

        return next_h, next_t, next_r

    def write_tfrecord(self, infile, outfile, hparams):
        sample_num = 0
        impression_id_list = []
        writer = tf.python_io.TFRecordWriter(outfile)
        try:
            for items, labels, h_list, t_list, r_list, mem_len_list, impression_ids in self._load_data_from_batch(infile, hparams):
                sample_num += len(labels)
                impression_id_list.extend(impression_ids)
                items =  np.asarray(items)
                item_str = items.tostring()
                labels = np.asarray(labels, dtype=np.float32)
                label_str = labels.tostring()
                feature = {
                    'item': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[item_str])),
                    'label': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[label_str]))
                }
                for hop in range(hparams.n_hops):
                    tmp_h_list = np.asarray(h_list[hop])
                    tmp_h_list = tmp_h_list.tostring()
                    feature["memory_head_" + str(hop)] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tmp_h_list]))
                    tmp_r_list = np.asarray(r_list[hop])
                    tmp_r_list = tmp_r_list.tostring()
                    feature["memory_relation_" + str(hop)] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tmp_r_list]))
                    tmp_t_list = np.asarray(t_list[hop])
                    tmp_t_list = tmp_t_list.tostring()
                    feature["memory_tail_" + str(hop)] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tmp_t_list]))
                    # tmp_m_list = np.asarray(mem_mask_list[hop])
                    # tmp_m_list = tmp_m_list.tostring()
                    # feature["memory_mask_" + str(hop)] = tf.train.Feature(
                    #     bytes_list=tf.train.BytesList(value=[tmp_m_list]))
                    tmp_l_list = np.asarray(mem_len_list[hop])
                    tmp_l_list = tmp_l_list.tostring()
                    feature["memory_length_" + str(hop)] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tmp_l_list]))
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature=feature
                    )
                )
                serialized = example.SerializeToString()
                writer.write(serialized)
        except:
            raise ValueError('train data format must be mkr, for example: user\titem\tlabel')
        writer.close()
        return sample_num, impression_id_list

    def _read_user_dict(self, user_dict_file):
        user_dict = dict()
        with open(user_dict_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip().split("\t")
                if not line:
                    break
                user = int(line[0])
                clicks = [int(x) for x in line[1:]]
                user_dict[user] = clicks
        return user_dict

    def _read_kg(self, kg_file):
        kg_dict = dict()
        with open(kg_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip().split("\t")
                if not line:
                    break
                head = int(line[0])
                tail = [int(x) for x in line[1].split(",")]
                relation = [int(x) for x in line[2].split(",")]
                kg_dict[head] = [tail, relation]
        return kg_dict