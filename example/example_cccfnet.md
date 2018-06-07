# Example: CCCFNet

## step 1. Prepare the data in the required format (cccfnet for CCCFNet)
The training data should be prepared in the required format as described in [/data_format/cccfnet_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/cccfnet_format.md).
We provide an example in [/data/cccfnet](https://github.com/zhfzhmsra/DeepRec/tree/master/data/cccfnet).

## step 2. Edit the configuration file and then run:
```
cp config/cccfnet_classfy.yaml config/network.yaml
```


## step 3. Train the model
```
python mainArg.py cccfnet train
```

## step 4. Infer the result
Configure which trained model you would like to use for inference in config/network.yaml, and then run:
```
python mainArg.py cccfnet infer
```