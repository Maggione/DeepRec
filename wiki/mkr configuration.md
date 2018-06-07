nework.yaml for Multi-task learning approach for Knowledge graph enhanced Recommendation for classification

```
#data
#data_format:dkn
data:
    train_file  :  data/mkr/ratings_final.txt => 训练数据
    eval_file  :  data/mkr/ratings_final.txt => 验证数据
    test_file  :  data/mkr/ratings_final.txt => 测试数据
    infer_file  :  data/mkr/ratings_final.txt =>预测数据
    kg_file  : data/mkr/kg_1_final.txt =>知识图谱文件
    n_users  : 6036 => 用户数目
    n_items  : 2355 => item数目
    n_entity : 146648 => 实体数目
    n_relation  : 20 => 关系数目
    data_format : mkr => mkr模型使用的数据格式名称

#model
#model_type:dkn
model:
    method : classification =>分类问题
    model_type : mkr =>模型名称
    dim : 100 =>word_embedding层的维度
    activation : [tanh] =>每层使用的激活函数

#train
#init_method: normal,tnormal,uniform,he_normal,he_uniform,xavier_normal,xavier_uniform
train:
    init_method: uniform ==>参数初始化方式
    init_value : 0.1 ==>参数初始化方差
    embed_l2 : 0.0000 ==>embedding层参数的l2正则
    embed_l1 : 0.0000 ==>embedding层参数的l1正则
    layer_l2 : 0.0000 ==>layer层参数的l2正则
    layer_l1 : 0.0000 0.0000 ==>layer层参数的l1正则
    lr_rs : 1 =>推荐部分学习率
    lr_kg : 0.5 =>图谱部分学习率
    loss : log_loss ==>logloss误差
    optimizer : sgd ==>sgd优化器
    epochs : 2 ==>总共训练5个epoch
    batch_size : 100 ==>batch_size选择15

#show info
#metric :'auc','logloss', 'group_auc', 'f1', 'logloss', 'acc'
info:
    show_step : 10 ==>每10个step, print一次信息
    save_epoch : 2 ==>每2个epoch，保存一次模型
    metrics : ['auc', 'logloss', 'acc', 'f1'] ==>使用的metric集合
```