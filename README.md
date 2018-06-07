<img src="https://s1.ax1x.com/2017/09/24/QzeaQ.png" width="509" height="176" />

# **Table Contents**
- **What is DeepRec** 
- **How to Use**
- **Benchmark Results**
- **References**

## **What is DeepRec**

DeepRec is a portable, flexible and comprehensive library including a variety of state-of-the-art deep learning based recommendation models. It aims to solve the item ranking task. In current version, DeepRec supports two kinds of methods: feature-based methods and knowledge-enhanced methods. In feature-based methods, deep learning models are applied to the extracted feature files with the specified format. In knowledge-enhanced methods, the signals from knowledge graph are leveraged to improve the recommendation performance. Current supported models are listed in the following, more methods will be expected in the near future. 

### **currently supported models**

model | type | data format | configuration example |
:---|:-----| :---|:------| 
lr | feature-based | [/data_format/libffm_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/libffm_format.md) | [/config/lr.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/lr.yaml) |
fm | feature-based | [/data_format/libffm_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/libffm_format.md) | [/config/fm.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/fm.yaml) |  
dnn | feature-based | [/data_format/libffm_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/libffm_format.md) | [/config/dnn.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/dnn.yaml) | 
[ipnn](https://arxiv.org/pdf/1611.00144.pdf) | feature-based | [/data_format/libffm_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/libffm_format.md) | [/config/ipnn.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/ipnn.yaml) | 
[opnn](https://arxiv.org/pdf/1611.00144.pdf) | feature-based | [/data_format/libffm_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/libffm_format.md) | [/config/opnn.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/opnn.yaml) | 
[deepWide](https://arxiv.org/abs/1606.07792) | feature-based | [/data_format/libffm_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/libffm_format.md) | [/config/deepWide.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/deepWide.yaml) |
[deepFM](https://arxiv.org/abs/1703.04247) | feature-based | [/data_format/libffm_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/libffm_format.md) | [/config/deepFM.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/deppFM.yaml) |
[deep&cross](https://arxiv.org/pdf/1708.05123.pdf) | feature-based |[/data_format/libffm_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/libffm_format.md) | [/config/deepcross.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/deepcross.yaml) |
[din](https://arxiv.org/pdf/1706.06978.pdf) | feature-based | [/data_format/din_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/din_format.md) | [/config/din.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/din.yaml) |
[cccfnet](https://dl.acm.org/citation.cfm?id=3054207) | feature-based | [/data_format/cccfnet_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/cccfnet_format.md) | [/config/cccfnet_classfy.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/cccfnet_classfy.yaml), [cccfnet_regress.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/cccfnet_regress.yaml) |
[dkn](https://dl.acm.org/citation.cfm?doid=3178876.3186175) | knowledge-enhanced | [/data_format/dkn_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/dkn_format.md) | [/config/dkn.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/dkn.yaml) |
exDeepFM | feature-based | [/data_format/libffm_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/libffm_format.md) | [/config/exDeepFM.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/exDeepFM.yaml)  |
[ripple](https://arxiv.org/abs/1803.03467) | knowledge-enhanced | [/data_format/ripple_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/ripple_format.md) | [/config/ripple.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/ripple.yaml) |
mkr | knowledge-enhanced | [/data_format/mkr_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/mkr_format.md) | [/config/mkr.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/mkr.yaml) |

<div align="center">Table 1</div>

## **How to Use**

#### **Requirement**
- Enviroment: linux, python 3 
- Dependent packages: tensorflow (>=1.4.0), sklearn, yaml, numpy 

#### **Usage**
  1. For each method, prepare your data as the corresponding format listed in Table 1.
  2. To set the parameters for your method, edit the corresponding configuration file listed in Table 1. Edit [network.yaml](https://github.com/zhfzhmsra/DeepRec/tree/master/config/network.yaml) based on your demand, please do not modify the name of the configuration file
  3. Run this kind of command "python mainArg.py [the choosed model name] train/infer"

#### **Examples**
  Here, we give the example of running ExDeepFM, more examples can be found [here]().
    1. Download the data and Prepare the data in the required format (Libffm for ExDeepFM)
    ```
cd data
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
python ML-100K2Libffm.py
```

## **Benchmark Results**
**Note**：because our tool is for the mulit-hot data type, that is more common. sparse matrix is ​​used to store data. building a network requires a lot of sparse operations. our tools are currently only for academic experiments, if the number of samples is larger than 1000w, and feature num is larger than 100w, our tool performance may be relatively low.
we are trying to improve efficiency 
### **benchmark-1**
we sample 300w from criteo dataset([dataset](https://www.kaggle.com/c/criteo-display-ad-challenge)), dealing with long tail features and continuous features. the dataset has 26w features and 300w samples.we split the dataset randomly into three parts: 80% is for training, 10% is for validating, 10% is for testing.


model | auc | logloss | train time per epoch/s|
----|------|------|------| 
lr | 0.7779 | 0.4692 | 20.4| 
fm | 0.7895 | 0.4591 | 90.8 |   
dnn | 0.7939 | 0.4552 | 425.1 |  
ipnn | 0.7947 | 0.4546 | 413.3 |  
opnn | 0.7957 | 0.4539 | 417.6 |  
deepWide | 0.7936 | 0.4557 | 412.4 | 
deepFM | 0.7944 | 0.4549 | 680.8 | 

### **benchmark-2**
we conduct experiment on Company* dataset.the dataset has 20w samples and 19w features. 

model | auc | logloss | train time per epoch/s|
----|------|------|------| 
lr | 0.6555 | 0.3914 | 21.9| 
fm | 0.6873 | 0.39 | 58.4 |   
dnn | 0.7315 | 0.3711 | 201.7 |  
ipnn | 0.7297 | 0.3712 | 199.3 |  
opnn | 0.7332 | 0.3698 | 197.3 |  
deepWide | 0.7346 | 0.3721 | 202.1 | 
deepFM | 0.7324 | 0.3759 | 233.6 | 
din | 0.7401 | 0.3763 | 331.4 | 

## References
- [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170)
- [DKN: Deep Knowledge-Aware Network for News Recommendation](https://arxiv.org/pdf/1801.08284v1.pdf)
- [A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)
- [Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/abs/1601.02376)
- [Product-based Neural Networks for User Response Prediction](https://arxiv.org/abs/1611.00144)
- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- [A Content-Boosted Collaborative Filtering Neural Network for Cross Domain Recommender Systems](http://dl.acm.org/citation.cfm?id=3054207)
- [product-nets](https://github.com/Atomu2014/product-nets)
- [RippleNetwork: Propagating User Preferences on the Knowledge Graph for RecommenderSystems](https://arxiv.org/pdf/1803.03467.pdf)