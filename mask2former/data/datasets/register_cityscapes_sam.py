# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

dataroot = '/cpfs01/projects-HDD/pujianxiangmuzu_HDD/public/mr'
annpath = f'mask2former/datasets/Cityscapes/train_gt.txt'
def cityscapes():
    # assert mode in ('train', 'eval', 'test')

    with open(annpath, 'r') as fr:
        pairs = fr.read().splitlines()
    img_paths, lb_paths = [], []
    for pair in pairs:
        imgpth, lbpth = pair.split(',')
        img_paths.append(osp.join(dataroot, imgpth))
        lb_paths.append(osp.join(dataroot, lbpth))

    assert len(img_paths) == len(lb_paths)
    dataset_dicts = []
    for (img_path, gt_path) in zip(img_paths, lb_paths):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts


def register_cityscapes():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    name = 'train'
    # dirname = 'train'
    
    name = f"cs_sem_seg_train"
    DatasetCatalog.register(
        name, cityscapes
    )
    MetadataCatalog.get(name).set(
        # stuff_classes=meta["stuff_classes"][:],
        # image_root=image_dir,
        # sem_seg_root=gt_dir,
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_cityscapes()
