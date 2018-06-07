din format is designed for deep interest model

```
0 News#3:392:1 News#3:1739:1 1:3427:1 News#4:3711:1 News#4:4055:1 News#4:4387:1 News#4:4897:1 News#4:6470:1 <br>
News#4:8160:1 News#4:8161:1 News#4:9020:1 News#4:9100:1 News#4:12311:1 News#4:12925:1 News#4:13638:1 <br>
News#4:13918:1 News#4:14453:1 User#4:36046:0 User#4:48386:0 User#4:86928:1 <br> 
2:110183:0.3 User#3:117161:1 User#3:117394:0 ...
```

格式说明：
数据其实分成两个部分，一部分field是要构建attention网络，一部分field不需要。 
因为PAIR_NUM = 4，所以，需要构建attention神经网络的field数为4个，一般field的数目为25。 
4个要构建attention的pair field使用标识符去标志。User端的4个field为[User#1,User#2 ,User#3, User#4]，news端的4个field为[News#1 ,News#2 ,News#3 , News#4]。
代码中，会根据标识符去判断该field是否要加入attention网络。Attention field和non-attention field的feat是一起进行编码的，共享一个embedding层。

din format 是为了deep interest network model(din)所设计的格式

tool和该模型相关的code和数据

类目 | 说明 | 
----|------|
example/din.yaml | 配置文件的例子 | 
IO/din_cache.py | 对din数据格式进行解析,压缩成tfrecord |   
IO/iterator.py | 读取缓存中之后的tfrecord数据进行训练 | 
src/din.py | 实现模型deep interest network | 
train.py | 训练模型的主程序 | 
data/din/train.attention.toy.txt,val.attention.toy.txt, test.attention.toy.txt | Deep interest network使用的toy 数据 |