# Towards Fewer Labels: Support Pair Active Learning for Person Re-identification (SPAL)
[Paper](https://arxiv.org/abs/2204.10008)


## Requirements

### Installation
```
git clone https://github.com/dapeng/SPAL.git
cd SPAL
python setup.py develop
```

### Prepare Datasets
```
cd examples && mkdir data
```
Download the person datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565). Then unzip them under the directory like
```
SPAL/examples/data
├── market1501
│   └── Market-1501-v15.09.15
└── msmt17
    └── MSMT17_V1
```

### Prepare ImageNet Pre-trained Models
- ImageNet-pretrained models for **ResNet-50** will be automatically downloaded in the python script.

- When training with the backbone of [IBN-ResNet](https://arxiv.org/abs/1807.09441), you need to download the ImageNet-pretrained model from this [link](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) and save it under the path of ```logs/pretrained/```.
  ```
  mkdir logs && cd logs
  mkdir pretrained
  ```
  The file tree should be
  ```
  SPAL/logs 
  └── pretrained
      └── resnet50_ibn_a.pth.tar
  ```

### Training
We utilize 4 GPUs for training. To train the model(s) in the paper, run this command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -d $DATASET --logs-dir $PATH_OF_LOGS
```
**Note that**
- We set the labeling budget to ```2n``` for all the experiments. The actual labeling cost is reported if the cost is less than the budget.

### Evaluation
To evaluate the model, run:
```
CUDA_VISIBLE_DEVICES=0 python examples/test.py -d $DATASET --resume $PATH
```


## Citation
If you find this code useful for your research, please cite our paper
```
@article{jin2022towards,
  title={Towards Fewer Labels: Support Pair Active Learning for Person Re-identification},
  author={Jin, Dapeng and Li, Minxian},
  journal={arXiv preprint arXiv:2204.10008},
  year={2022}
}
```


## Acknowledgements
Thanks to Yixiao Ge for opening source of her excellent works [SpCL](https://github.com/yxgeee/SpCL).
