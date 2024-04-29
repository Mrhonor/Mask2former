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
        stuff_classes= ['caravan', ' Pedestrian Area', ' Crosswalk', '', ' Curb', ' Parking', ' Tunnel', ' Barrier', ' Bike Lane', ' Bike Rack or bannister', ' Street Light', ' Trash Can', ' train', ' Other Vehicle or autorickshaw', ' crt screen', ' Wheeled Slow or swivel chair', ' Ego Vehicle', ' fruit', ' Ground Animal', ' Crosswalk - Plain', ' tie', ' Snow', ' broccoli', ' fork', ' Fire Hydrant', ' gravel', ' Traffic Sign (Back) or cardboard', ' Car Mount', '', ' obs-str-bar-fallback', ' bookshelf', ' picture', ' desk', ' clothes', ' lamp', ' bathtub', ' flag or handbag', ' window-blind', ' paper', ' wall-wood', ' Utility Pole', ' grass', ' house', ' sea', ' rug', ' field', ' rock', ' base', ' dirt-merged', ' river', ' coffee table', ' flower', ' bench', ' stove', ' light', ' airplane', ' escalator or wall-brick', ' bottle', ' cell phone', ' plaything or net', ' basket or cup', ' tent', ' bag', ' ball', ' food-other-merged', ' microwave', ' potted plant', ' vase', ' plate', ' monitor', ' glass', ' clock', ' terrain', ' Junction Box', ' Rail Track', ' billboard', ' Bridge', ' Mountain', ' Sand', ' Water', ' Boat', ' floor', ' cabinet', ' bed', ' sofa', ' table', ' door', ' window', ' counter', ' shelves', ' curtain', ' pillow', ' mirror', ' ceiling', ' books', ' television', ' towel', ' toilet', '', ' chair', ' Catch Basin or sink', ' Phone Booth or refridgerator', ' Guard Rail', ' sidewalk', ' pole', ' vegetation', ' traffic sign', ' road', ' building', ' fence', ' traffic light', ' sky', ' car', ' truck', ' bus', ' motorcycle', ' bicycle', ' wall', ' person', ' Service Lane', ' Bicyclist', ' rider', ' Lane Marking - General', ' Manhole', ' Traffic Sign Frame', '', '', '', ' earth', ' armchair', ' seat', ' dresser', ' cushion', ' column', ' signboard', ' chest of drawers', ' skyscraper', ' fireplace', ' grandstand', ' runway', ' case', ' pool table', ' screen door', ' stairway', ' blinds', ' hill', ' countertop', ' palm', ' kitchen island', ' arcade machine', ' hovel', ' tower', ' chandelier', ' awning', ' booth', ' dirt track', ' land', ' ottoman or snowboard', ' buffet', ' poster', ' stage', ' ship', ' fountain', ' conveyer belt', ' canopy', ' washer', ' swimming pool', ' stool', ' barrel', ' waterfall', ' cradle', ' oven', ' tank', '', ' animal or baseball bat', ' lake', ' dishwasher', ' whiteboard or screen', ' blanket', ' sculpture', ' hood', ' sconce', ' fan or suitcase', ' pier', '  Billboard', ' shower', ' radiator', '  parking meter', '  bird', '  cat', '  horse', '  sheep', '  cow', '  elephant', '  bear', '  zebra', '  giraffe', '  umbrella', '', ' kite', '', ' baseball glove', ' skateboard', ' surfboard', ' tennis racket', ' knife', ' spoon', ' bowl', ' banana', ' apple', ' sandwich', ' orange', ' carrot', ' hot dog', ' pizza', ' donut', ' cake', ' mouse', ' remote', ' keyboard', ' toaster', ' scissors', '', ' toothbrush', ' blanket', ' floor-wood', ' platform', ' roof', ' wall-tile', ' pavement-merged', ' rock-merged'],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        stuff_colors=[[254, 205,  75],[178, 109,  61],     [232, 182, 126],     [109,  64,  73],     [ 66,  69, 176],     [109, 220, 239],     [ 18,  58,  98],     [212, 221, 176],     [230, 246,  34],     [141,  17,  60],     [ 26, 184,  83],     [ 30,  47, 184],     [106, 248, 185],     [125, 123, 180],     [ 80, 108,  57],     [230, 175, 199],     [214,  92,  19],     [251, 225,  86],     [ 50, 105,  96],     [  0,  81, 175],     [113, 216, 102],     [148, 126,  90],     [ 30, 123,  47],     [208, 211,  35],     [ 73, 208,  97],     [100,  32,  22],     [169, 102, 174],     [131, 218,  87],     [224, 216, 151],     [160, 111,   0],     [163, 186, 153],     [141, 251,  82],     [ 83,  69,  70],     [112,  78, 139],     [168,  62, 253],     [ 12,  71,  15],     [164, 121, 219],     [177,  96,  58],     [230, 101, 208],     [252,   9, 152],     [ 68,  68, 237],     [188,  49,  31],     [189, 161,  55],     [199, 183,  46],     [195, 177, 180],     [112,   2, 104],     [234,  44,  84],     [ 25, 111,  21],     [ 39, 192,  19],     [136,  66,  69],     [ 91, 182,  84],     [218,  37, 136],     [ 47, 118, 136],     [109, 248,  25],     [104, 162, 189],     [ 87,  59,  76],     [176,  92, 116],     [217, 135,  68],     [249, 125, 105],     [231, 117, 185],     [ 99,  16, 132],     [189, 124,   9],     [108, 156,  92],     [ 31, 148,  91],     [180,  12, 242],     [184, 102,  69],     [206, 106,  97],     [111, 173,  82],     [213, 208, 155],     [167, 118,  46],     [158,  92,  94],     [115, 177,  95],     [155, 117, 111],     [ 58,  34,   9],     [  9,  60, 225],     [215,  48,  75],     [ 76,  66, 237],     [252, 168, 166],     [135, 161, 131],     [ 79,  24, 196],     [106,  48, 148],     [ 23,  69, 195],     [233, 198,  64],     [ 10,   7, 111],     [ 75, 211,  41],     [  7, 162,   5],     [ 77, 110,  58],     [ 35, 133,  44],     [138,  37,  98],     [126,  22, 239],     [113,  39, 205],     [ 54,   3, 109],     [146, 240, 187],     [ 19, 228,  89],     [240,  79,  15],     [ 61, 206, 227],     [235,  91,  37],     [109, 155, 209],     [ 24, 192, 253],     [ 38, 181, 228],     [236, 223, 130],     [177, 158, 120],     [180, 241, 180],     [236,   2, 131],     [ 70,  94, 120],     [128, 109,  69],     [218, 118, 115],     [ 97, 187, 159],     [154, 172, 132],     [145, 208,  42],     [153,  29, 134],     [225, 115,  77],     [ 14, 159,  57],     [ 81, 255, 160],     [ 79, 127, 244],     [ 12,  50, 106],     [103,  81, 234],     [174, 203, 111],     [198, 221, 182],     [195, 172, 234],     [199, 215,  29],     [ 64, 129, 196],     [226,  27,  43],     [ 34, 168, 123],     [186, 140, 216],     [126, 101, 165],     [ 77, 252, 183],     [ 34, 155,  59],     [ 25, 250, 252],     [132, 178,  25],     [214, 111, 212],     [167, 230, 156],     [ 56,  31, 101],     [192,  49, 133],     [225,  30,  12],     [191,  27, 123],     [ 18,  18, 139],     [211, 209, 237],     [ 55, 204,   1],     [ 38,  98, 154],     [ 51,  56,  95],     [ 72, 172, 147],     [137, 139, 117],     [127, 244, 110],     [233,  94, 191],     [ 87,  48, 117],     [127,   2,  56],     [220, 218,  77],     [219, 103,  62],     [ 87, 177, 247],     [193, 106,  86],     [ 33, 116, 176],     [  3, 186, 105],     [174,  67, 252],     [142, 222, 139],     [240,  29, 224],     [149, 202, 142],     [ 68, 217,   9],     [171, 130, 103],     [110, 237, 116],     [233,  37,  88],     [171,  98,  21],     [174,  68,   3],     [240, 255, 149],     [139,  23,  73],     [214,   3, 173],     [249, 159, 243],     [114,  64,  69],     [248, 107, 184],     [172, 222,  36],     [178, 135,   6],     [193, 163, 215],     [  9,  57, 229],     [ 44,  87,  68],     [ 76,  21,  14],     [164, 161,  80],     [230, 183, 128],     [195, 211, 185],     [ 31, 228, 121],     [235,  96,  62],     [198, 132,  42],     [ 46, 140, 152],     [184,  49, 182],     [129, 139, 126],     [ 27, 138, 140],     [146,  38,  43],     [ 89,  54, 145],     [157, 128,  24],     [189,  47,  88],     [133, 110, 199],     [164, 226,  74],     [ 79, 252,  40],     [ 99,  99, 147],     [160, 116,  78],     [ 71, 235,  99],     [208,  49, 185],     [193, 200, 137],     [254,  47,  11],     [ 33,   4, 237],     [201,  31, 122],     [ 16, 176,  44],     [152, 133, 134],     [ 84,  79, 104],     [121, 137, 142],     [109,  36, 216],     [129,  86, 148],     [177,  40,  46],     [ 16, 226,  71],     [ 15, 161,  47],     [ 39,  59, 205],     [138, 195,  40],     [ 57,  14, 178],     [153, 109, 201],     [ 27,  45, 173],     [154,  81, 177],     [205, 251, 130],     [  0, 247,  36],     [177, 163, 165],     [230, 232, 157],     [ 15, 236, 141],     [129, 125, 231],     [ 55, 211,  67],     [146,  99,  68],     [ 99, 244,  92],     [158, 123, 213],     [236, 135, 127],     [223, 218,  93],     [175, 119, 215],     [230, 197,  29],     [182,  60, 104],     [126,  91,  65]],
        evaluator_type="sem_seg",
        ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
    )



# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_uni()
