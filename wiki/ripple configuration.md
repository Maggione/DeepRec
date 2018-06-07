ripple.yaml for RippleNetwork model

you can find more configuration in example
``` 
#data
#data format:ripple
data:
     train_file  :  data/ripple/train.txt => 训练数据路径
     eval_file  :  data/ripple/val.txt => 验证数据路径
     test_file  :  data/ripple/test.txt =>测试数据路径     	
     infer_file  :  data/ripple/infer.txt =>预测数据路径
     kg_file  : data/ripple/kg.txt =>知识图谱数据路径
     user_clicks  : data/ripple/user_click.txt =>用户点击数据路径
     n_entity : 79125 =>实体数目
     n_memory : 459 =>每一跳包含的实体最大数目
     n_relation : 25 =>实体关系数目
     entity_limit : 10 =>每个实体的下一跳关联的最大数目
     user_click_limit : 30 =>用户历史点击的最大数目
     data_format : ripple => ripple数据格式

#model
#model_type:deepFM or deepWide or dnn or ipnn or opnn or fm or lr
model:
    method : classification => 分类问题
    model_type : ripple => RippleNetwork模型
    n_entity_emb : 50 => embedding size of entities (vector) 
    n_relation_emb : 50 => embedding size of relations (2-D matrix)
    is_use_relation : True
    dtype : 32 => embedding value type(tf.float16, tf.float32, tf.float64)
    n_hops: 2 => the number of ripple
    item_update_mode : "map_item" => update item mode, ["plus", "map_o",  "map_item", "map_all"]
    predict_mode : "inner_product" => the method to compute the output final probabilities, ["MLP", "inner_product", "DCN"]
    n_DCN_layer : 2  => when user "DCN" do prediction, the layer number
    is_map_feature : False
    n_map_emb : 50  => the dimension of the feature after multiply a transformation matrix
    is_clip_norm : False  => decide if clip the grad norm
    max_grad_norm : 10  => maximum grad norm when doing clipping
    kg_ratio : 1.0  => the ratio of the kgs number of the first hop
    reg_kg : 0.1 => regularization
    output_using_all_hops : True
    activation : [linear] => 网络结构中各层使用的激活函数
    
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