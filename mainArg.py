"""This script parse and run train function"""
import train
import utils.util as util
import tensorflow as tf
from utils.log import Log
import sys
import os


def flat_config(config):
    """flat config to a dict"""
    f_config = {}
    category = ['data', 'model', 'train', 'info']
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config


def create_hparams(FLAGS):
    """Create hparams."""
    FLAGS = flat_config(FLAGS)
    return tf.contrib.training.HParams(
        # data
        train_file=FLAGS['train_file'] if 'train_file' in FLAGS else None,
        eval_file=FLAGS['eval_file'] if 'eval_file' in FLAGS else None,
        test_file=FLAGS['test_file'] if 'test_file' in FLAGS else None,
        infer_file=FLAGS['infer_file'] if 'infer_file' in FLAGS else None,
        kg_file=FLAGS['kg_file'] if 'kg_file' in FLAGS else None,
        user_clicks=FLAGS['user_clicks'] if 'user_clicks' in FLAGS else None,
        FEATURE_COUNT=FLAGS['FEATURE_COUNT'] if 'FEATURE_COUNT' in FLAGS else None,
        FIELD_COUNT=FLAGS['FIELD_COUNT'] if 'FIELD_COUNT' in FLAGS else None,
        data_format=FLAGS['data_format'] if 'data_format' in FLAGS else None,
        PAIR_NUM=FLAGS['PAIR_NUM'] if 'PAIR_NUM' in FLAGS else None,
        DNN_FIELD_NUM=FLAGS['DNN_FIELD_NUM'] if 'DNN_FIELD_NUM' in FLAGS else None,
        n_user=FLAGS['n_user'] if 'n_user' in FLAGS else None,
        n_item=FLAGS['n_item'] if 'n_item' in FLAGS else None,
        n_user_attr=FLAGS['n_user_attr'] if 'n_user_attr' in FLAGS else None,
        n_item_attr=FLAGS['n_item_attr'] if 'n_item_attr' in FLAGS else None,
        ### ripple
        n_entity = FLAGS['n_entity'] if 'n_entity' in FLAGS else None,
        n_memory = FLAGS['n_memory'] if 'n_memory' in FLAGS else None,
        n_relation = FLAGS['n_relation'] if 'n_relation' in FLAGS else None,
        n_users = FLAGS['n_users'] if 'n_users' in FLAGS else None,
        n_items = FLAGS['n_items'] if 'n_items' in FLAGS else None,
        entity_limit = FLAGS['entity_limit'] if 'entity_limit' in FLAGS else None,
        user_click_limit = FLAGS['user_click_limit'] if 'user_click_limit' in FLAGS else None,
        # dkn
        wordEmb_file=FLAGS['wordEmb_file'] if 'wordEmb_file' in FLAGS else None,
        entityEmb_file=FLAGS['entityEmb_file'] if 'entityEmb_file' in FLAGS else None,
        doc_size=FLAGS['doc_size'] if 'doc_size' in FLAGS else None,
        word_size=FLAGS['word_size'] if 'word_size' in FLAGS else None,
        entity_size=FLAGS['entity_size'] if 'entity_size' in FLAGS else None,
        entity_dim=FLAGS['entity_dim'] if 'entity_dim' in FLAGS else None,
        entity_embedding_method=FLAGS['entity_embedding_method']
        if 'entity_embedding_method' in FLAGS else None,
        transform=FLAGS['transform'] if 'transform' in FLAGS else None,
        train_ratio=FLAGS['train_ratio'] if 'train_ratio' in FLAGS else None,

        # model
        dim=FLAGS['dim'] if 'dim' in FLAGS else None,
        layer_sizes=FLAGS['layer_sizes'] if 'layer_sizes' in FLAGS else None,
        cross_layer_sizes=FLAGS['cross_layer_sizes'] if 'cross_layer_sizes' in FLAGS else None,
        cross_layers=FLAGS['cross_layers'] if 'cross_layers' in FLAGS else None,
        activation=FLAGS['activation'] if 'activation' in FLAGS else None,
        cross_activation=FLAGS['cross_activation'] if 'cross_activation' in FLAGS else "identity",
        dropout=FLAGS['dropout'] if 'dropout' in FLAGS else [0.0],
        attention_layer_sizes=FLAGS['attention_layer_sizes'] if 'attention_layer_sizes' in FLAGS else None,
        attention_activation=FLAGS['attention_activation'] if 'attention_activation' in FLAGS else None,
        attention_dropout=FLAGS['attention_dropout'] \
            if 'attention_dropout' in FLAGS else 0.0,
        model_type=FLAGS['model_type'] if 'model_type' in FLAGS else None,
        method=FLAGS['method'] if 'method' in FLAGS else None,
        load_model_name= os.path.join(util.MODEL_DIR, FLAGS['load_model_name']) if 'load_model_name' in FLAGS else None,
        infer_model_name=os.path.join(util.MODEL_DIR, FLAGS['infer_model_name']) if 'infer_model_name' in FLAGS else None,
        filter_sizes=FLAGS['filter_sizes'] if 'filter_sizes' in FLAGS else None,
        num_filters=FLAGS['num_filters'] if 'num_filters' in FLAGS else None,
        mu=FLAGS['mu'] if 'mu' in FLAGS else None,
        ###ripple
        is_use_relation = FLAGS["is_use_relation"] if "is_use_relation" in FLAGS else False,
        n_entity_emb=FLAGS["n_entity_emb"] if "n_entity_emb" in FLAGS else None,
        n_relation_emb=FLAGS["n_relation_emb"] if "n_relation_emb" in FLAGS else None,
        n_map_emb=FLAGS["n_map_emb"] if "n_map_emb" in FLAGS else None,
        n_hops=FLAGS["n_hops"] if "n_hops" in FLAGS else None,
        item_update_mode=FLAGS["update_item_embedding"] if "update_item_embedding" in FLAGS else None,
        predict_mode=FLAGS["predict_mode"] if "predict_mode" in FLAGS else None,
        n_DCN_layer=FLAGS["n_DCN_layer"] if "n_DCN_layer" in FLAGS else None,
        is_map_feature=FLAGS["is_map_feature"] if "is_map_feature" in FLAGS else False,
        kg_ratio=FLAGS["kg_ratio"] if "kg_ratio" in FLAGS else 1.0,
        output_using_all_hops =FLAGS["output_using_all_hops"] if "output_using_all_hops" in FLAGS else False,

        # train
        init_method=FLAGS['init_method'] if 'init_method' in FLAGS else 'tnormal',
        init_value=FLAGS['init_value'] if 'init_value' in FLAGS else 0.01,
        embed_l2=FLAGS['embed_l2'] if 'embed_l2' in FLAGS else 0.0000,
        embed_l1=FLAGS['embed_l1'] if 'embed_l1' in FLAGS else 0.0000,
        layer_l2=FLAGS['layer_l2'] if 'layer_l2' in FLAGS else 0.0000,
        layer_l1=FLAGS['layer_l1'] if 'layer_l1' in FLAGS else 0.0000,
        cross_l2=FLAGS['cross_l2'] if 'cross_l2' in FLAGS else 0.0000,
        cross_l1=FLAGS['cross_l1'] if 'cross_l1' in FLAGS else 0.0000,
        reg_kg=FLAGS["reg_kg"] if "reg_kg" in FLAGS else 0.0000,
        learning_rate=FLAGS['learning_rate'] if 'learning_rate' in FLAGS else 0.001,
        lr_rs=FLAGS["lr_rs"] if 'lr_rs' in FLAGS else 1,
        lr_kg=FLAGS["lr_kg"] if 'lr_kg' in FLAGS else 0.5,
        kg_training_interval=FLAGS["kg_training_interval"] if 'kg_training_interval' in FLAGS else 5,
        max_grad_norm = FLAGS['max_grad_norm'] if 'max_grad_norm' in FLAGS else 2,
        is_clip_norm = FLAGS['is_clip_norm'] if 'is_clip_norm' in FLAGS else 0,
        dtype = FLAGS['dtype'] if 'dtype' in FLAGS else 32,
        loss=FLAGS['loss'] if 'loss' in FLAGS else None,
        optimizer=FLAGS['optimizer'] if 'optimizer' in FLAGS else 'adam',
        epochs=FLAGS['epochs'] if 'epochs' in FLAGS else 10,
        batch_size=FLAGS['batch_size'] if 'batch_size' in FLAGS else 1,
        # show info
        show_step=FLAGS['show_step'] if 'show_step' in FLAGS else 1,
        save_epoch=FLAGS['save_epoch'] if 'save_epoch' in FLAGS else 5,
        metrics=FLAGS['metrics'] if 'metrics' in FLAGS else None
    )


def check_type(config):
    """check config type"""
    # check parameter type
    int_parameters = ['word_size', 'entity_size', 'doc_size', 'FEATURE_COUNT', 'FIELD_COUNT', 'dim', 'epochs', 'batch_size', 'show_step', \
                      'save_epoch', 'PAIR_NUM', 'DNN_FIELD_NUM', 'attention_layer_sizes', \
                      'n_user', 'n_item', 'n_user_attr', 'n_item_attr']
    for param in int_parameters:
        if param in config and not isinstance(config[param], int):
            raise TypeError("parameters {0} must be int".format(param))

    float_parameters = ['init_value', 'learning_rate', 'embed_l2', \
                        'embed_l1', 'layer_l2', 'layer_l1', 'mu']
    for param in float_parameters:
        if param in config and not isinstance(config[param], float):
            raise TypeError("parameters {0} must be float".format(param))

    str_parameters = ['train_file', 'eval_file', 'test_file', 'infer_file', 'method', \
                      'load_model_name', 'infer_model_name', 'loss', 'optimizer', 'init_method', 'attention_activation']
    for param in str_parameters:
        if param in config and not isinstance(config[param], str):
            raise TypeError("parameters {0} must be str".format(param))

    list_parameters = ['layer_sizes', 'activation', 'dropout']
    for param in list_parameters:
        if param in config and not isinstance(config[param], list):
            raise TypeError("parameters {0} must be list".format(param))

    if ('data_format' in config) and (not config['data_format'] in ['ffm', 'din', 'cccfnet', 'dkn', 'ripple', 'mkr']):
        raise TypeError("parameters data_format must be din" \
                        ",ffm, cccfnet, dkn, ripple but is {0}".format(config['data_format']))



def check_nn_config(config):
    """check neural networks config"""
    if config['model']['model_type'] in ['fm']:
        required_parameters = ['train_file', 'eval_file', 'FEATURE_COUNT', 'dim', 'loss', 'data_format', 'method']
    elif config['model']['model_type'] in ['lr']:
        required_parameters = ['train_file', 'eval_file', 'FEATURE_COUNT', 'loss', 'data_format', 'method']
    elif config['model']['model_type'] in ['din']:
        required_parameters = ['train_file', 'eval_file', 'PAIR_NUM', 'DNN_FIELD_NUM', 'FEATURE_COUNT', 'dim', \
                               'layer_sizes', 'activation', 'attention_layer_sizes', 'attention_activation', 'loss', \
                               'data_format', 'dropout', 'method']
    elif config['model']['model_type'] in ['cccfnet']:
        required_parameters = ['train_file', 'eval_file', 'dim', 'layer_sizes', 'n_user', 'n_item', 'n_user_attr',
                               'n_item_attr',
                               'activation', 'loss', 'data_format', 'dropout', 'mu', 'method']
    elif config['model']['model_type'] in ['dkn']:
        required_parameters = ['doc_size', 'train_file', 'eval_file', 'wordEmb_file', 'entityEmb_file',
                               'word_size', 'entity_size', 'data_format', 'dim', 'layer_sizes', 'activation',
                               'attention_activation', 'attention_activation', 'attention_dropout', 'loss', \
                               'data_format', 'dropout', 'method', 'num_filters', 'filter_sizes']
    elif config['model']['model_type'] in ['exDeepFM']:
        required_parameters = ['train_file', 'eval_file', 'FIELD_COUNT', 'FEATURE_COUNT', 'method',
                               'dim', 'layer_sizes', 'cross_layer_sizes', 'activation', 'loss', 'data_format', 'dropout']
    elif config['model']['model_type'] in ['CIN']:
        required_parameters = ['train_file', 'eval_file', 'FIELD_COUNT', 'FEATURE_COUNT', 'method',
                               'dim', 'cross_layer_sizes', 'loss', 'data_format']
    elif config['model']['model_type'] in ['deepcross']:
        required_parameters = ['train_file', 'eval_file', 'FIELD_COUNT', 'FEATURE_COUNT', 'method',
                               'dim', 'layer_sizes', 'cross_layers', 'activation', 'loss', 'data_format', 'dropout']
    elif config['model']['model_type'] in ['ripple']:
        required_parameters = ['train_file', 'eval_file', 'kg_file', 'user_clicks', 'n_entity', 'n_memory', 'n_relation', 'entity_limit', 'user_click_limit',
                               'n_entity_emb', 'n_relation_emb', 'n_hops', 'item_update_mode', 'predict_mode', 'n_DCN_layer', 'n_map_emb']
    elif config['model']['model_type'] in ['mkr']:
        required_parameters = ['train_file', 'eval_file', 'kg_file', 'n_entity',
                               'n_relation', 'n_users', 'n_items', 'dim', 'activation']
    else:
        required_parameters = ['train_file', 'eval_file', 'FIELD_COUNT', 'FEATURE_COUNT', 'method', \
                               'dim', 'layer_sizes', 'activation', 'loss', 'data_format', 'dropout']
    f_config = flat_config(config)
    # check required parameters
    for param in required_parameters:
        if param not in f_config:
            raise ValueError("parameters {0} must be set".format(param))
    if f_config['model_type'] == 'din':
        if f_config['data_format'] != 'din':
            raise ValueError(
                "for din model, data format must be din, but your set is {0}".format(f_config['data_format']))
    elif f_config['model_type'] == 'cccfnet':
        if f_config['data_format'] != 'cccfnet':
            raise ValueError(
                "for cccfnet model, data format must be cccfnet, but your set is {0}".format(f_config['data_format']))
    elif f_config['model_type'] == 'dkn':
        if f_config['data_format'] != 'dkn':
            raise ValueError(
                "for dkn model, data format must be dkn, but your set is {0}".format(f_config['data_format']))
    elif f_config["model_type"] == 'ripple':
        if f_config['data_format'] != 'ripple':
            raise ValueError(
                "for ripple model, data format must be ripple, but your set is {0}".format(f_config['data_format']))
    elif f_config["model_type"] == 'mkr':
        if f_config['data_format'] != 'mkr':
            raise ValueError(
                "for mkr model, data format must be mkr, but your set is {0}".format(f_config['data_format']))
    else:
        if f_config['data_format'] != 'ffm':
            raise ValueError("data format must be ffm, but your set is {0}".format(f_config['data_format']))
    check_type(f_config)


def check_config(config):
    """check networks config"""
    if config['model']['model_type'] not in ['deepFM', 'deepWide', 'dnn', 'ipnn', \
                                             'opnn', 'fm', 'lr', 'din', 'cccfnet', 'dkn', 'deepcross', 'exDeepFM', "CIN", 'ripple', 'mkr']:
        raise ValueError(
            "model type must be cccfnet, deepFM, deepWide, dnn, ipnn, opnn, fm, lr, din, dkn, deepcross, exDeepFM, CIN, ripple but you set is {0}".format(
                config['model']['model_type']))
    check_nn_config(config)


# train process load yaml
def load_yaml():
    """load config from yaml"""
    yaml_name = util.CONFIG_DIR + util.TRAIN_YAML
    print('training network configuration file is {0}'.format(yaml_name))
    util.check_file_exist(yaml_name)
    config = util.load_yaml_file(yaml_name)
    return config


def main():
    """main function"""
    if len(sys.argv)>1:
        util.USER_ID = sys.argv[1]
        def mk_for_userid():
            util.CACHE_DIR = os.path.join(util.CACHE_DIR, util.USER_ID) + '/'
            util.RES_DIR = os.path.join(util.RES_DIR, util.USER_ID) + '/'
            util.LOG_DIR = os.path.join(util.LOG_DIR, util.USER_ID) + '/'
            util.MODEL_DIR = os.path.join(util.MODEL_DIR, util.USER_ID) + '/'
            util.SUMMARIES_DIR = os.path.join(util.SUMMARIES_DIR, util.USER_ID) + '/'
            # util.TRAIN_YAML = util.USER_ID + ".yaml"
        mk_for_userid()
        if len(sys.argv) > 2:
            util.INFER = sys.argv[2]
    # flag = True
    util.check_tensorflow_version()
    util.check_and_mkdir()
    config = load_yaml()
    check_config(config)
    hparams = create_hparams(config)
    hparams = create_hparams(config)
    log = Log(hparams)
    hparams.logger = log.logger
    print(hparams.values())
    import train
    train.train(hparams)


if __name__  == "__main__":
    main()
