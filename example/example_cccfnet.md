# Example: CCCFNet

## step 1. Prepare the data in the required format (cccfnet for CCCFNet)
The training data should be prepared in the required format as described in [wiki/cccfnet format.md](https://deeprec.visualstudio.com/deeprec/_wiki/wikis/deeprec.wiki?wikiVersion=GBwikiMaster&pagePath=%2FDeepRec%2FCCCFNet%20Format).
We provide an example in data/cccfnet.

## step 2. Edit the configuration file
```
cp example/cccfnet_classfy.yaml config/network.yaml
```
For detail, see the introduction about the parameter setting in [wiki/cccfnet configuration.md](https://deeprec.visualstudio.com/deeprec/_wiki/wikis/deeprec.wiki?wikiVersion=GBwikiMaster&pagePath=%2FDeepRec%2FCCCFNet%20Configuration).

## step 3. Train the model
```
python mainArg.py cccfnet train
```
The first argv element is the directory name for the results. For example, it will create cache/cccfnet directory to save your cache file, 
checkpoint/cccfnet to save your trained model, logs/cccfnet to save your training log.

The second argv element is about the mode. If you want to train a model, you choose "train". If you want to infer results, you choose "infer".

## step 4. Infer the result
Configure which trained model you would like to use for inference in config/network.yaml, and then run:
```
python mainArg.py cccfnet infer
```