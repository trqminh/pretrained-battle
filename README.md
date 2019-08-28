# pretrained-battle

### description
in this repo, I have written and refactored some code to compare performance of feature extract pretrained models
in different frameworks.

### references
https://github.com/colingogo/caffe-pretrained-feature-extraction

### run code
requirements:
- cuda 8.0 + caffe 1.0
- cuda 9.0 + pytorch 1.1.0
...


ROOT = path/to/clone-repo/


##### prepare dataset
- download dogs vs cats dataset from: https://www.kaggle.com/c/dogs-vs-cats and put it in ~/Downloads/

```sh
cd ROOT
mkdir dataset
mkdir dogs-vs-cats
```

- create train.txt and test.txt like this repo: https://github.com/colingogo/caffe-pretrained-feature-extraction

##### for caffe:
###### 1. download some files

```sh
cd ROOT
cd caffe
```

download imagenet_mean.binaryproto

```sh
cd pretrained_models
```

download alexnet.caffemodel
download vgg16.caffemodel
download googlenet.caffemodel

###### 2. run

```python
cd ROOT/caffe
python convert_to_npy.py
mkdir features
python get_features.py
```

##### for pytorch:
in progress