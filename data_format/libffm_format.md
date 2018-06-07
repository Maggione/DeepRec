format description is as [here](https://github.com/guestwalk/libffm)

libffm格式的数据适用于nn-based的方法,包括deepFM, deepWide, dnn, ipnn, opnn, fm, lr

tool和该模型相关的code和数据

code | 说明 |
----|------|
config/deepFM.yaml,deepWide.yaml,dnn.yaml,fm.yaml,ipnn.yaml,lr.yaml,opnn.yaml | 配置文件的例子 |
IO/ffm_cache.py | 对ffm数据格式进行解析,压缩成tfrecord |
IO/iterator.py | 配置文件的例子 |
IO/ffm_cache.py | 读取缓存中之后的tfrecord数据进行训练 |
src/deep_fm.py, deep_wide.py, dnn.py, fm.py, ipnn.py, lr.py, opnn.py | 模型deepFM, deepWide, dnn, fm, ipnn, opnn, lr |
train.py | 训练模型的主程序 |
data/dnn/train.userid.txt, val.userid.txt, test.userid.txt, infer.userid.txt | NN类模型使用的toy 数据 |