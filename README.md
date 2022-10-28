# YOLOv7_OBB
## The code for the implementation of “[Yolov7](https://arxiv.org/abs/2207.02696) + [Kullback-Leibler Divergence](https://link.zhihu.com/?target=https%3A//ieeexplore.ieee.org/abstract/document/9852282/)”. 
## Results and Models
To be continued
## Installation
### Requirements
* Windows (Recommend), Linux (Recommend)
* Python 3.7+ 
* PyTorch ≥ 1.7 
* CUDA 9.0 or higher


### Install 
**CUDA Driver Version ≥ CUDA Toolkit Version(runtime version) = torch.version.cuda**

a. Create a conda virtual environment and activate it, e.g.,
```
conda create -n Py39_Torch1.10_cu11.3 python=3.9 -y 
source activate Py39_Torch1.10_cu11.3
```
b. Make sure your CUDA runtime api version ≤ CUDA driver version. (for example 11.3 ≤ 11.4)
```
nvcc -V
nvidia-smi
```
c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), Make sure cudatoolkit version same as CUDA runtime api version, e.g.,
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
nvcc -V
python
>>> import torch
>>> torch.version.cuda
>>> exit()
```
d. Clone the yolov7_obb repository.
```
git clone https://github.com/lx-cly/YOLOv7_OBB.git
cd YOLOv7_OBB
```
e. Install yolov7_obb(like yolov7).

```python 
pip install -r requirements.txt
cd utils/nms_rotated
python setup.py develop  #or "pip install -v -e ."
```

### Install DOTA_devkit. (only in Windows) 
**(Custom Install, it's just a tool to split the high resolution image and evaluation the obb)**

```
cd YOLOv7_OBB/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

### Prepare dataset
```
parent
├── yolov7
└── datasets
    └── DOTAv1.5
        ├── train_split_rate1.0_subsize1024_gap200
        ├── train_split_rate1.0_subsize1024_gap200
        └── test_split_rate1.0_subsize1024_gap200
            ├── images
            └── labelTxt

```

**Note:**

* DOTA is a high resolution image dataset, so it must be splited before training to get a better performance.

## Train a model

**1. Prepare custom dataset files**

1.1 Make sure the labels format is [poly classname diffcult], e.g., 
```
  x1      y1       x2        y2       x3       y3       x4       y4       classname     diffcult

1686.0   1517.0   1695.0   1511.0   1711.0   1535.0   1700.0   1541.0   large-vehicle      1
```
**(*Note: You can set **diffcult=0**)**
![image](https://user-images.githubusercontent.com/72599120/159213229-b7c2fc5c-b140-4f10-9af8-2cbc405b0cd3.png)


1.2 Split the dataset. 
```shell
cd YOLOv7_OBB
python DOTA_devkit/ImgSplit_multi_process.py
```
or Use the orignal dataset. 
```shell
cd YOLOv7_OBB
```
**(*Note: High resolution image dataset needs to be splited to get better performance in small objects)**


**2. Train**

2.1 Train with specified GPUs. (for example with GPU=3)

```shell
python train.py --device 3
```

2.2 Train with multiple(4) GPUs. (DDP Mode)

```shell
python -m torch.distributed.launch --nproc_per_node 4 train.py --device 0,1,2,3
```

## Inferenece with pretrained models. (Splited Dataset)
This repo provides the validation/testing scripts to evaluate the trained model.

Examples:

Assume that you have already downloaded the checkpoints to `runs/train/exp/weights`.

1. Test YOLOv7_OBB with single GPU. Get the HBB metrics.

```shell
python test.py --task 'val' --device 0 --save-json --batch-size 2 --data 'data/demo_split.yaml' --name 'obb_demo_split'

2. Parse the results. Get the poly format results.
```shell 
python tools/TestJson2VocClassTxt.py --json_path 'runs/val/obb_demo_split/best_obb_predictions.json' --save_path 'runs/val/obb_demo_split/obb_predictions_Txt'
```

3. Merge the results. (If you split your dataset )
```shell
python DOTA_devkit/ResultMerge_multi_process.py \
    --scrpath 'runs/val/obb_demo_split/obb_predictions_Txt' \
    --dstpath 'runs/val/obb_demo_split/obb_predictions_Txt_Merged'
```

4. Get the OBB metrics
```shell
python DOTA_devkit/dota_evaluation_task1.py \
    --detpath 'runs/val/obb_demo_split/obb_predictions_Txt_Merged/Task1_{:s}.txt' \
    --annopath 'dataset/dataset_demo/labelTxt/{:s}.txt' \
    --imagesetfile 'dataset/dataset_demo/imgnamefile.txt'

## Run inference on images, videos, directories, streams, etc. Then save the detection file.
1. image demo
```shell
python detect.py --weights 'runs/train/exp/weights/best.pt' \
    --source 'dataset/dataset_demo/images/' \
    --img 1024 --device 2 --hide-labels --hide-conf
```

##  Acknowledgements
I have used utility functions from other wonderful open-source projects. Espeicially thank the authors of:

* [ultralytics/yolov5](https://github.com/ultralytics/yolov5).
* [open-mmlab/mmrotate](https://github.com/open-mmlab/mmrotate).
* [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [hukaixuan19970627/yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb)
* [CAPTAIN-WHU/DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
