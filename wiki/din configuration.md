nework.yaml for deep interest network(din) for classification 
``` 
#data
#data format:ffm
data:
    train_file  :  data/din/train.attention.toy.txt => 训练数据
    eval_file  :  data/din/val.attention.toy.txt => 验证数据
    test_file  :  data/din/test.attention.toy.txt => 测试数据
    #infer_file  :  data/cccfnet/test.txt =>预测数据
    PAIR_NUM :  4 => 需要使用attention的field的pair数
    DNN_FIELD_NUM : 25 => 一般的field的数目
    FEATURE_COUNT : 194081 => feat的总数,attention field feat和一般的field feat 共享embedding矩阵,因此feat一起编号
    data_format : din => din模型使用的数据格式名称

#model
#model_type:deepFM or deepWide or dnn or ipnn or opnn or fm or lr
model:
    method : classification =>分类问题
    model_type : din =>模型名称
    dim : 10 =>embedding层的维度
    layer_sizes : [100, 100]=>网络结构
    activation : [relu, relu]=>每层使用的激活函数
    dropout : [0.5, 0.5]=>每层的dropout
    attention_layer_sizes : 10=>attention网络隐层的结点数
    attention_activation : relu=>attention网络隐层的激活函数
    #load_model_name : ./checkpoint/epoch_2


#train
#init_method: normal,tnormal,uniform,he_normal,he_uniform,xavier_normal,xavier_uniform
train:
    init_method: tnormal ==>参数初始化方式
    init_value : 0.1 ==>参数初始化方差
    embed_l2 : 0.0000 ==>embedding层参数的l2正则
    embed_l1 : 0.0000 ==>embedding层参数的l1正则
    layer_l2 : 0.0000 ==>layer层参数的l2正则
    layer_l1 : 0.0000 0.0000 ==>layer层参数的l1正则
    learning_rate : 0.0001 ==>学习率
    loss : log_loss ==>logloss误差
    optimizer : adam ==>adam优化器
    epochs : 5 ==>总共训练5个epoch
    batch_size : 15 ==>batch_size选择15

#show info
#metric :'auc','logloss', 'group_auc'
info:
    show_step : 10 ==>每10个step, print一次信息
    save_epoch : 2 ==>每2个epoch，保存一次模型
    metrics : ['auc', 'logloss'] ==>使用的metric集合
```