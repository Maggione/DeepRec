nework.yaml for deep knowledge-aware network(dkn) for classification
```
#data
#data_format:dkn
data:
    train_file  :  data/dkn/final_train_with_entity.txt => 训练数据
    eval_file  :  data/dkn/final_test_with_entity.txt => 验证数据
    test_file  :  data/dkn/final_test_with_entity.txt => 测试数据
    #infer_file  :  data/dkn/final_test_with_entity.txt =>预测数据
    wordEmb_file : data/dkn/word_embeddings_100.npy => pre-trained word embedding
    entityEmb_file : data/dkn/TransE_entity2vec_100.npy => pre-trained entity embedding (must have the same dims as the configuration "dim" in this file)
    doc_size  :  10 =>每个特征的词个数
    word_size  :  8033 =>单词词典大小
    entity_size  ：  3777 =>实体词典大小
    data_format : dkn => dkn模型使用的数据格式名称

#model
#model_type:dkn
model:
    method : classification =>分类问题
    model_type : dkn =>模型名称
    dim : 100 =>word_embedding层的维度
    entity_dim : 100 =>entity_embedding层的维度
    entity_embedding_method : TransE =>关系抽取方式
    transform : true =>对entity_embedding做映射，和Word_embedding同一个空间
    layer_sizes : [300] =>网络结构
    activation : [sigmoid] =>每层使用的激活函数
    dropout : [0.0] =>每层的dropout
    filter_sizes : [1,2,3] =>kcnn卷积核大小
    num_filters : 100 =>卷积核个数
    attention_layer_sizes : 100 =>attention网络隐层的结点数
    attention_activation : relu =>attention网络隐层的激活函数
    attention_dropout : 0.0 =>attention网络隐层的dropout
    #load_model_name : ./checkpoint/epoch_2

#train
#init_method: normal,tnormal,uniform,he_normal,he_uniform,xavier_normal,xavier_uniform
train:
    init_method: uniform ==>参数初始化方式
    init_value : 0.1 ==>参数初始化方差
    embed_l2 : 0.0000 ==>embedding层参数的l2正则
    embed_l1 : 0.0000 ==>embedding层参数的l1正则
    layer_l2 : 0.0000 ==>layer层参数的l2正则
    layer_l1 : 0.0000 0.0000 ==>layer层参数的l1正则
    learning_rate : 0.0001 ==>学习率
    loss : log_loss ==>logloss误差
    optimizer : adam ==>adam优化器
    epochs : 2 ==>总共训练5个epoch
    batch_size : 100 ==>batch_size选择15

#show info
#metric :'auc','logloss', 'group_auc', 'f1', 'logloss', 'acc'
info:
    show_step : 10 ==>每10个step, print一次信息
    save_epoch : 2 ==>每2个epoch，保存一次模型
    metrics : ['auc', 'logloss', 'acc', 'f1'] ==>使用的metric集合
```