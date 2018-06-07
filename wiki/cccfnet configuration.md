nework.yaml for cccfnet for regression 
``` 
#data
#data format:ffm
data:
     train_file  :  data/cccfnet/train.txt ==>训练数据路径
     eval_file  :  data/cccfnet/eval.txt ==>验证数据路径
     test_file  :  data/cccfnet/test.txt ==>测试数据路径
     #infer_file  :                        =>预测数据路径                       
     n_user : 671 ==> 数据集中user的数量
     n_item : 9066 ==> 数据集中item的数量
     n_user_attr : 764 ==> 数据集中user属性的数量
     n_item_attr : 24 ==> 数据集中item属性的数量
     data_format : cccfnet ==> cccfnet特有的数据格式的名称
     mu : 3.543579578399658 ==>cccfnet有矩阵分解的部分，需要加上全局的偏置，mu可以通过统计训练集中的平均分得到

#model
#model_type:deepFM or deepWide or dnn or ipnn or opnn or fm or lr cccfnet
#method:classification and regression
model:
    method : regression ==> tool中实现的是评分问题，即regression 
    model_type : cccfnet ==> cccfnet在tool中的模型简称
    dim : 16 ==>embedding的维度
    layer_sizes : [16] ==> 提取attribute embedding使用的网络的结构
    activation : [sigmoid] ==> 网络结构中各层使用的激活函数
    dropout : [0.0] ==> 网络结构中各层使用的dropout
#    load_model_name : ./checkpoint/epoch_1


#train
#init_method: normal,tnormal,uniform,he_normal,he_uniform,xavier_normal,xavier_uniform
train:
    init_method: tnormal ==>参数初始化方式
    init_value : 0.01 ==>参数初始化方差
    embed_l2 : 0.000 ==>embedding层参数的l2正则
    embed_l1 : 0.0000 ==>embedding层参数的l1正则
    layer_l2 : 0.000 ==>layer层参数的l2正则
    layer_l1 : 0.0000 ==>layer层参数的l1正则
    learning_rate : 0.1 ==>学习率
    loss : square_loss ==>最小均分误差
    optimizer : sgd ==>sgd优化器
    epochs : 10 ==>总共训练10个epoch
    batch_size : 500 ==>batch_size选择500

#show info
#metric :'auc','logloss', 'rmse', 'group_auc'
info:
    show_step : 10 ==>每10个step, print一次信息
    save_epoch : 2 ==>每2个epoch，保存一次模型
    metrics : ['rmse'] ==>使用的metric集合

```

nework.yaml for cccfnet for classification
``` 
#data
#data format:ffm
data:
     train_file  :  data/cccfnet/train_c.txt
     eval_file  :  data/cccfnet/eval_c.txt
     test_file  :  data/cccfnet/test_c.txt
     #infer_file  :  data/infer.entertainment.no_inter.norm.fieldwise.userid.txt
     n_user : 671
     n_item : 9066
     n_user_attr : 764
     n_item_attr : 24
     data_format : cccfnet
     mu : 0.821944177093359

#model
#model_type:deepFM or deepWide or dnn or ipnn or opnn or fm or lr cccfnet
#method:classification and regression
model:
    method : classification
    model_type : cccfnet
    dim : 16
    layer_sizes : [16]
    activation : [sigmoid]
    dropout : [0.0]
#    load_model_name : ./checkpoint/epoch_1


#train
#init_method: normal,tnormal,uniform,he_normal,he_uniform,xavier_normal,xavier_uniform
train:
    init_method: tnormal
    init_value : 0.01
    embed_l2 : 0.000
    embed_l1 : 0.0000
    layer_l2 : 0.000
    layer_l1 : 0.0000
    learning_rate : 0.1
    loss : log_loss
    optimizer : sgd
    epochs : 10
    batch_size : 500

#show info
#metric :'auc','logloss', 'rmse', 'group_auc'
info:
    show_step : 10
    save_epoch : 2
    metrics : ['auc', 'logloss']

```