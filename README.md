# A Simple Explicit Quality Network for Face Recognition

## Training Data

1. Download [MS1Mv2](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
2. Extract image files by [rec2image.py](https://github.com/deepinsight/insightface/blob/master/recognition/common/rec2image.py)
3. Generate the training file list
```
cd dataset
python generate_file_list.py
```

## Training
1. Step 1: set config.py, then run **python train_feature.py**
```
```
2. Step 2: set config.py, then run **python train_quality.py**
```
```
3. Step 3: set config.py, then run **python train_feature.py**
```
```