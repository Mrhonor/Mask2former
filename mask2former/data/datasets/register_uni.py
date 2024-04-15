# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp
import numpy as np

dataroot = '/cpfs01/projects-HDD/pujianxiangmuzu_HDD/public/mr/ADEChallengeData2016'
annpath = f'mask2former/datasets/ADE/validation.txt'
def uni():
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

palette = np.random.randint(0, 256, (512, 3), dtype=np.uint8)
palette[0] = [153, 153, 153]
palette[94] = [220, 220, 0]
palette[98] = [128, 64, 128]
palette[103] = [70, 70, 70]
palette[104] = [190, 153, 153]
palette[106] = [107, 142, 35]
palette[107] = [70, 130, 180]
palette[108] = [0, 0, 142]
palette[97] = [220, 20, 60]
palette[102] = [244, 35, 232]



def register_uni():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for i in range(268):
        lb_map[i] = i
        # if el['id'] == 0:
        #     lb_map[el['id']] = 255
        # else:
        #     lb_map[el['id']] = el['id'] - 1

    name = f"uni"
    DatasetCatalog.register(
        name, uni
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes=[str(i) for i in range(268)],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        stuff_colors=palette,
        evaluator_type="sem_seg",
        ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
    )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_uni()
