# Fine-Grained Image Classification

Final project of the course, to train a deep learning model for fine-grained image classification on dataset [CUB_200_2011](http://www.vision.caltech.edu/datasets/cub_200_2011/).

This is a group project, and other contributors are *Xin Hao* and *Yiming Wang*.

Use [TransFG](https://arxiv.org/abs/2103.07976) as our model.

## Work Requirement

- Source code and trained model
- A README file
- A report, consisting of introduction to the model, the validation result and references.
- Slides for in-class presentation
- (Optional) Design a GUI, which shows the classification and wiki knowledge of the item in the input image

## Usage

### Train

Download [CUB dataset](http://www.vision.caltech.edu/datasets/cub_200_2011/
) and [ViT-B_16 model](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false).

Arrange the files as below. 

```
TransFG
    ├─ dataset
    │       └─ CUB_200_2011
    ├─ models
    ├─ utils
    ├─ train.py
    └─ ViT-B_16.npz
```

Directly run `train.py`.

### Show demo

Download our [pretrained model](https://disk.pku.edu.cn:443/link/F65B24FCC4B01D2A4F7352D2A73D3DC1).

Arrange the files as below.

```
TransFG
    ├─ pedia
    ├─ models
    ├─ utils
    ├─ default_checkpoint.bin
    └─ demo.py
```

Directly run `demo.py`
