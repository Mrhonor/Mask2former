## Getting Started with Mask2Former

This document provides a brief intro of the usage of Mask2Former.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.


### Inference Demo with Pre-trained Models

1. Pick a model and its config file from
  [model zoo](MODEL_ZOO.md),
  for example, `configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml`.
2. We provide `demo.py` that is able to demo builtin configs. Run it with:
```
python demo/demo.py --config-file ../configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --input input1.jpg  \
  --output output/         \
  --opts MODEL.WEIGHTS /path/to/checkpoint_file
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.


### Training & Evaluation in Command Line

We provide a script `train_net.py`, that is made to train all the configs provided in Mask2Former.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](./datasets/README.md),
then run:
```
python train_net.py --num-gpus 8 \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml
```

The configs are made for 8-GPU training.
Since we use ADAMW optimizer, it is not clear how to scale learning rate with batch size.
To train on 1 GPU, you need to figure out learning rate and batch size by yourself:
```
python train_net.py \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE
```

To evaluate a model's performance, use
```
python train_net.py \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `python train_net.py -h`.


### Video instance segmentation
Please use `demo_video/demo.py` for video instance segmentation demo and `train_net_video.py` to train
and evaluate video instance segmentation models.
