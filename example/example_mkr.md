# Example: MKR

## step 1. Prepare the data in the required format (mkr for MKR) 
Two kinds of data is required: knowledge graph, user-item pairs. They should be prepared in the required format as described in [/data_format/mkr_format.md](https://github.com/zhfzhmsra/DeepRec/tree/master/data_format/mkr_format.md).
We provide an example in [/data/mkr](https://github.com/zhfzhmsra/DeepRec/tree/master/data/mkr).

## step 2. Edit the configuration file and then run:
```
cp config/mkr.yaml config/network.yaml
```


## step 3. Train the model
```
python mainArg.py mkr train
```

## step 4. Infer the result
```
python mainArg.py mkr infer
```