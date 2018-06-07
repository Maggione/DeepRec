nework.yaml for dnn-based model(deepFM, deepWide, dnn, ipnn, opnn, lr, fm) for classification

you can find more configuration in example
``` 
#data
#data format:ffm
data:
     train_file  :  data/dnn/train.userid.txt => 训练数据路径
     eval_file  :  data/dnn/val.userid.txt => 验证数据路径
     test_file  :  data/dnn/test.userid.txt =>测试数据路径     	
     infer_file  :  data/dnn/infer.userid.txt =>预测数据路径
     FIELD_COUNT :  33 => field的数据,libffm数据的格式
     FEATURE_COUNT : 194081 => 特征的数量
     data_format : ffm => ffm数据格式

#model
#model_type:deepFM or deepWide or dnn or ipnn or opnn or fm or lr
model:
    method : classification => 分类问题
    model_type : deepFM => deepFM模型
    dim : 10 =>embedding的维度
    layer_sizes : [100, 100] => NN使用的网络的结构
    activation : [relu, relu] => 网络结构中各层使用的激活函数
    dropout : [0.5, 0.5] => 网络结构中各层使用的dropout
#    load_model_name : ./checkpoint/epoch_1


#train
#init_method: normal,tnormal,uniform,he_normal,he_uniform,xavier_normal,xavier_uniform
train:
    init_method: tnormal ==>参数初始化方式
    init_value : 0.1 ==>参数初始化方差
    embed_l2 : 0.001 ==>embedding层参数的l2正则
    embed_l1 : 0.0000 ==>embedding层参数的l1正则
    layer_l2 : 0.001 ==>layer层参数的l2正则
    layer_l1 : 0.0000 ==>layer层参数的l1正则
    learning_rate : 0.0001 ==>学习率
    loss : log_loss ==> log_loss
    optimizer : adam ==>adam优化器
    epochs : 2 ==>总共训练2个epoch
    batch_size : 3 ==>batch_size选择3

#show info
#metric :'auc','logloss', 'group_auc'
info:
    show_step : 1 ==>每1个step, print一次信息
    save_epoch : 2 ==>每2个epoch，保存一次模型
    metrics : ['auc','logloss']==>使用的metric集合

```