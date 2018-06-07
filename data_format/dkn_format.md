dkn format is designed for deep knowledge-aware network

```
0 CandidateNews:1050,5238,5165,6996,1922,7058,7263,0,0,0 entity:0,0,0,0,2562,2562,2562,0,0,0
clickedNews0:7680,3978,1451,409,4681,0,0,0,0,0 entity0:0,0,0,395,0,0,0,0,0,0
clickedNews1:3698,2301,1055,6481,6242,7203,0,0,0,0 entity1:0,0,1904,1904,1904,0,0,0,0,0 ...
```

格式说明：
dkn模型主要针对的是文本内容的推荐，包括新闻推荐（输入为候选新闻和用户的历史点击新闻）
每一条样本的格式如下：
[label] [CandiateNews:w1,w2,w3,...] [entity:e1,e2,e3,...] [clickedNews0:w1,w2,w3,...] [entity1:e1,e2,e3,...] ...
1、每一项之间用' '(空格)隔开，第一项是样本标签；
2、候选新闻特征的格式为：'CandiateNews:'+候选新闻中的单词编号，每个单词之间用','隔开；'entity:'+候选新闻中的对应单词的实体编号，每个单词之间用','隔开，若无则补0；
3、用户历史点击新闻特征的格式为：'clickNews#:'+候选新闻中的单词编号，每个单词之间用','隔开；'entity#:'+候选新闻中的对应单词的实体编号，每个单词之间用','隔开；其中#代表每条点击的编号。
4、每条新闻截取的单词个数必须一致，缺少部分补0。


tool和该模型相关的code和数据

类目 | 说明 | 
----|------|
config/dkn.yaml | 配置文件的例子 |
IO/dkn_cache.py | 对din数据格式进行解析,压缩成tfrecord |
IO/iterator.py | 读取缓存中之后的tfrecord数据进行训练 | 
src/dkn.py | 实现模型deep knowledge-aware network |
train.py | 训练模型的主程序 | 
data/dkn/final_train_with_entity.txt,final_test_with_entity.txt | deep knowledge-aware network使用的数据 |
data/dkn/TransE_entity2vec_100.npy, word_embeddings_100.npy | deep knowledge-aware network使用的预先训练好的entity_embedding和word_embedding(文件命名格式是[entity_embedding_method]_entity2vec_[entity_dim].npy, word_embeddings_[dim].py)