# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import cv2
from collections import OrderedDict
from typing import Any, Dict, List, Set

import argparse
import torch
from PIL import Image
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    MaskFormerSemanticDatasetMapper_2,
    SemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    add_hrnet_config,
    add_gnn_config,
    DaLiLoaderAdapter,
    LoaderAdapter,
    build_bipartite_graph_for_unseen,
    build_bipartite_graph_for_unseen_for_manually,
    eval_for_mseg_datasets,
    find_specific_class
)

from PIL import Image
from detectron2.utils.file_io import PathManager
import numpy as np
from functools import partial
from detectron2.structures import ImageList
import torch.nn.functional as F
import logging

from mask2former.utils.evaluate import eval_link_hook, iter_info_hook


logger = logging.getLogger(__name__)
def my_sem_seg_loading_fn(filename, dtype=int, lb_map=None, size_divisibility=-1, ignore_label=255):
    with PathManager.open(filename, "rb") as f:
        image = np.array(Image.open(f), copy=False, dtype=dtype)
        if lb_map is not None:
            image = lb_map[image] 

    #     logger.info(f'size_divisibility: {size_divisibility}')
    #     if size_divisibility > 0:
    #         image = torch.tensor(image)
            
    #         image_size = (image.shape[0], image.shape[1])
    #         padding_size = [
    #             0,
    #             size_divisibility - image_size[1],
    #             0,
    #             size_divisibility - image_size[0],
    #         ]
            
    #         image = F.pad(image, padding_size, value=ignore_label).contiguous()
    #         logger.info(f'image shape: {image.shape}')
    #         image = image.numpy()

    # dsaf
    return image
    

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_hrnet_config(cfg)
    add_gnn_config(cfg)
    # add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

def build_bipart_for_unseen(cfg, model):
    """
    Build bipartite graph for unseen classes.
    """
    from mask2former.utils import build_bipartite_graph_for_unseen
    build_bipartite_graph_for_unseen(cfg, model)
    

def main(args):
    # cfg = setup(args)
    
    # if args.eval_only:
    dataset_name = 'mapi_sem_seg_train'
    meta = MetadataCatalog.get(dataset_name)
    lookup_table = MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id
    stuff_colors = MetadataCatalog.get(dataset_name).stuff_colors
    stuff_colors.extend([[0,0,0] for _ in range(256-len(stuff_colors))])
    stuff_colors = np.array(stuff_colors)
    lb_map = np.arange(256).astype(np.uint8)
    for k, v in lookup_table.items():
        lb_map[k] = v
    # f = '/home1/marong/datasets/mapi/validation/labels/v1.2/Vy9DxMoNR2FtDgrwDXo-nA_L.png'
    f = args.input[0]
    # f = '/home1/marong/datasets/ADEChallengeData2016/annotations/validation/ADE_val_00000045.png'
    ori_image = np.array(Image.open(f), copy=False, dtype=int)
    image = lb_map[ori_image]
    color_image = stuff_colors[image] #image.point(lb_map)
    visualizer = Visualizer(color_image, meta, instance_mode=ColorMode.IMAGE)
    
    lb_map[44] = 255
    image = lb_map[ori_image]
    out = visualizer.draw_sem_seg(image)
    out.save('rgb_image.png')
    # cv2.imwrite('rgb_image.png', image[:,:,::-1])


    # 保存转换后的图片
    # rgb_image.save("rgb_image.jpg")
        # build_bipartite_graph_for_unseen_for_manually(Trainer.build_test_loader, cfg, model)
        # return
        # build_bipartite_graph_for_unseen(Trainer.build_test_loader, cfg, model)
        # res = Trainer.test(cfg, model)
        # if cfg.TEST.AUG.ENABLED:
        #     res.update(Trainer.test_with_TTA(cfg, model))
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        # return res
    
    # trainer = Trainer(cfg)
    # trainer.register_hooks([eval_link_hook(), iter_info_hook()])
    # # trainer.register_hooks([iter_info_hook()])
    # trainer.resume_or_load(resume=args.resume)
    # return trainer.train()

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
    # args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args, ),
    # )
