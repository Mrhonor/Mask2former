# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

labels_info = [
    {"name": "unlabel", "id": 0, "trainId": 255},
    {"name": "flag", "id": 150, "trainId": 0},
    {"name": "wall", "id": 1, "trainId": 1},
    {"name": "building", "id": 2, "trainId": 2},
    {"name": "sky", "id": 3, "trainId": 3},
    {"name": "floor", "id": 4, "trainId": 4},
    {"name": "tree", "id": 5, "trainId": 5},
    {"name": "ceiling", "id": 6, "trainId": 6},
    {"name": "road", "id": 7, "trainId": 7},
    {"name": "bed", "id": 8, "trainId": 8},
    {"name": "windowpane", "id": 9, "trainId": 9},
    {"name": "grass", "id": 10, "trainId": 10},
    {"name": "cabinet", "id": 11, "trainId": 11},
    {"name": "sidewalk", "id": 12, "trainId": 12},
    {"name": "person", "id": 13, "trainId": 13},
    {"name": "earth", "id": 14, "trainId": 14},
    {"name": "door", "id": 15, "trainId": 15},
    {"name": "table", "id": 16, "trainId": 16},
    {"name": "mountain", "id": 17, "trainId": 17},
    {"name": "plant", "id": 18, "trainId": 18},
    {"name": "curtain", "id": 19, "trainId": 19},
    {"name": "chair", "id": 20, "trainId": 20},
    {"name": "car", "id": 21, "trainId": 21},
    {"name": "water", "id": 22 , "trainId": 22 },
    {"name": "painting", "id": 23, "trainId": 23},
    {"name": "sofa", "id": 24 , "trainId": 24 },
    {"name": "shelf", "id": 25 , "trainId": 25 },
    {"name": "house", "id": 26 , "trainId": 26 },
    {"name": "sea", "id": 27 , "trainId": 27 },
    {"name": "mirror", "id": 28, "trainId": 28},
    {"name": "rug", "id": 29, "trainId": 29},
    {"name": "field", "id": 30, "trainId": 30},
    {"name": "armchair", "id": 31, "trainId": 31},
    {"name": "seat", "id": 32, "trainId": 32},
    {"name": "fence", "id": 33, "trainId": 33},
    {"name": "desk", "id": 34, "trainId": 34},
    {"name": "rock", "id": 35, "trainId": 35},
    {"name": "wardrobe, closet, press", "id": 36, "trainId": 36},
    {"name": "lamp", "id": 37, "trainId": 37},
    {"name": "bathtub, bathing tub, bath, tub", "id": 38, "trainId": 38},
    {"name": "railing, rail", "id": 39, "trainId": 39},
    {"name": "cushion", "id": 40, "trainId": 40},
    {"name": "base, pedestal, stand", "id": 41, "trainId": 41},
    {"name": "box", "id": 42, "trainId": 42},
    {"name": "column, pillar", "id": 43, "trainId": 43},
    {"name": "signboard, sign", "id": 44, "trainId": 44},
    {"name": "chest of drawers, chest, bureau, dresser", "id": 45, "trainId": 45},
    {"name": "counter", "id": 46, "trainId": 46},
    {"name": "sand", "id": 47, "trainId": 47},
    {"name": "sink", "id": 48, "trainId": 48},
    {"name": "skyscraper", "id": 49, "trainId": 49},
    {"name": "fireplace, hearth, open fireplace", "id": 50, "trainId": 50},
    {"name": "refrigerator, icebox", "id": 51, "trainId": 51},
    {"name": "grandstand, covered stand", "id": 52, "trainId": 52},
    {"name": "path", "id": 53, "trainId": 53},
    {"name": "stairs, steps", "id": 54, "trainId": 54},
    {"name": "runway", "id": 55, "trainId": 55},
    {"name": "case, display case, showcase, vitrine", "id": 56, "trainId": 56},
    {"name": "pool table, billiard table, snooker table", "id": 57, "trainId": 57},
    {"name": "pillow", "id": 58, "trainId": 58},
    {"name": "screen door, screen", "id": 59, "trainId": 59},
    {"name": "stairway, staircase", "id": 60, "trainId": 60},
    {"name": "river", "id": 61, "trainId": 61},
    {"name": "bridge, span", "id": 62, "trainId": 62},
    {"name": "bookcase", "id": 63, "trainId": 63},
    {"name": "blind, screen", "id": 64, "trainId": 64},
    {"name": "coffee table, cocktail table", "id": 65, "trainId": 65},
    {"name": "toilet, can, commode, crapper, pot, potty, stool, throne", "id": 66, "trainId": 66},
    {"name": "flower", "id": 67, "trainId": 67},
    {"name": "book", "id": 68, "trainId": 68},
    {"name": "hill", "id": 69, "trainId": 69},
    {"name": "bench", "id": 70, "trainId": 70},
    {"name": "countertop", "id": 71, "trainId": 71},
    {"name": "stove, kitchen stove, range, kitchen range, cooking stove", "id": 72, "trainId": 72},
    {"name": "palm, palm tree", "id": 73, "trainId": 73},
    {"name": "kitchen island", "id": 74, "trainId": 74},
    {"name": "computer, computing machine, computing device, data processor, electronic computer, information processing system", "id": 75, "trainId": 75},
    {"name": "swivel chair", "id": 76, "trainId": 76},
    {"name": "boat", "id": 77, "trainId": 77},
    {"name": "bar", "id": 78, "trainId": 78},
    {"name": "arcade machine", "id": 79, "trainId": 79},
    {"name": "hovel, hut, hutch, shack, shanty", "id": 80, "trainId": 80},
    {"name": "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", "id": 81, "trainId": 81},
    {"name": "towel", "id": 82, "trainId": 82},
    {"name": "light, light source", "id": 83, "trainId": 83},
    {"name": "truck, motortruck", "id": 84, "trainId": 84},
    {"name": "tower", "id": 85, "trainId": 85},
    {"name": "chandelier, pendant, pendent", "id": 86, "trainId": 86},
    {"name": "awning, sunshade, sunblind", "id": 87, "trainId": 87},
    {"name": "streetlight, street lamp", "id": 88, "trainId": 88},
    {"name": "booth, cubicle, stall, kiosk", "id": 89, "trainId": 89},
    {"name": "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box", "id": 90, "trainId": 90},
    {"name": "airplane, aeroplane, plane", "id": 91, "trainId": 91},
    {"name": "dirt track", "id": 92, "trainId": 92},
    {"name": "apparel, wearing apparel, dress, clothes", "id": 93, "trainId": 93},
    {"name": "pole", "id": 94, "trainId": 94},
    {"name": "land, ground, soil", "id": 95, "trainId": 95},
    {"name": "bannister, banister, balustrade, balusters, handrail", "id": 96, "trainId": 96},
    {"name": "escalator, moving staircase, moving stairway", "id": 97, "trainId": 97},
    {"name": "ottoman, pouf, pouffe, puff, hassock", "id": 98, "trainId": 98},
    {"name": "bottle", "id": 99, "trainId": 99},
    {"name": "buffet, counter, sideboard", "id": 100, "trainId": 100},
    {"name": "poster, posting, placard, notice, bill, card", "id": 101, "trainId": 101},
    {"name": "stage", "id": 102, "trainId": 102},
    {"name": "van", "id": 103, "trainId": 103},
    {"name": "ship", "id": 104, "trainId": 104},
    {"name": "fountain", "id": 105, "trainId": 105},
    {"name": "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "id": 106, "trainId": 106},
    {"name": "canopy", "id": 107, "trainId": 107},
    {"name": "washer, automatic washer, washing machine", "id": 108, "trainId": 108},
    {"name": "plaything, toy", "id": 109, "trainId": 109},
    {"name": "swimming pool, swimming bath, natatorium", "id": 110, "trainId": 110},
    {"name": "stool", "id": 111, "trainId": 111},
    {"name": "barrel, cask", "id": 112, "trainId": 112},
    {"name": "basket, handbasket", "id": 113, "trainId": 113},
    {"name": "waterfall, falls", "id": 114, "trainId": 114},
    {"name": "tent, collapsible shelter", "id": 115, "trainId": 115},
    {"name": "bag", "id": 116, "trainId": 116},
    {"name": "minibike, motorbike", "id": 117, "trainId": 117},
    {"name": "cradle", "id": 118, "trainId": 118},
    {"name": "oven", "id": 119, "trainId": 119},
    {"name": "ball", "id": 120, "trainId": 120},
    {"name": "food, solid food", "id": 121, "trainId": 121},
    {"name": "step, stair", "id": 122, "trainId": 122},
    {"name": "tank, storage tank", "id": 123, "trainId": 123},
    {"name": "trade name, brand name, brand, marque", "id": 124, "trainId": 124},
    {"name": "microwave, microwave oven", "id": 125, "trainId": 125},
    {"name": "pot, flowerpot", "id": 126, "trainId": 126},
    {"name": "animal, animate being, beast, brute, creature, fauna", "id": 127, "trainId": 127},
    {"name": "bicycle, bike, wheel, cycle ", "id": 128, "trainId": 128},
    {"name": "lake", "id": 129, "trainId": 129},
    {"name": "dishwasher, dish washer, dishwashing machine", "id": 130, "trainId": 130},
    {"name": "screen, silver screen, projection screen", "id": 131, "trainId": 131},
    {"name": "blanket, cover", "id": 132, "trainId": 132},
    {"name": "sculpture", "id": 133, "trainId": 133},
    {"name": "hood, exhaust hood", "id": 134, "trainId": 134},
    {"name": "sconce", "id": 135, "trainId": 135},
    {"name": "vase", "id": 136, "trainId": 136},
    {"name": "traffic light, traffic signal, stoplight", "id": 137, "trainId": 137},
    {"name": "tray", "id": 138, "trainId": 138},
    {"name": "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin", "id": 139, "trainId": 139},
    {"name": "fan", "id": 140, "trainId": 140},
    {"name": "pier, wharf, wharfage, dock", "id": 141, "trainId": 141},
    {"name": "crt screen", "id": 142, "trainId": 142},
    {"name": "plate", "id": 143, "trainId": 143},
    {"name": "monitor, monitoring device", "id": 144, "trainId": 144},
    {"name": "bulletin board, notice board", "id": 145, "trainId": 145},
    {"name": "shower", "id": 146, "trainId": 146},
    {"name": "radiator", "id": 147, "trainId": 147},
    {"name": "glass, drinking glass", "id": 148, "trainId": 148},
    {"name": "clock", "id": 149, "trainId": 149},
]
dataroot = '/home1/marong/datasets/ADEChallengeData2016'
annpath = f'mask2former/datasets/ADE/validation.txt'
def ade():
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


def register_ade():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']

    name = f"ade_sem_seg_val"
    DatasetCatalog.register(
        name, ade
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes=["flag", "wall", "building, edifice", "sky", "floor, flooring", "tree", "ceiling", "road, route", "bed ", "windowpane, window ", "grass", "cabinet", "sidewalk, pavement", "person, individual, someone, somebody, mortal, soul", "earth, ground", "door, double door", "table", "mountain, mount", "plant, flora, plant life", "curtain, drape, drapery, mantle, pall", "chair", "car, auto, automobile, machine, motorcar", "water", "painting, picture", "sofa, couch, lounge", "shelf", "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair", "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press", "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion", "base, pedestal, stand", "box", "column, pillar", "signboard, sign", "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink", "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox", "grandstand, covered stand", "path", "stairs, steps", "runway", "case, display case, showcase, vitrine", "pool table, billiard table, snooker table", "pillow", "screen door, screen", "stairway, staircase", "river", "bridge, span", "bookcase", "blind, screen", "coffee table, cocktail table", "toilet, can, commode, crapper, pot, potty, stool, throne", "flower", "book", "hill", "bench", "countertop", "stove, kitchen stove, range, kitchen range, cooking stove", "palm, palm tree", "kitchen island", "computer, computing machine, computing device, data processor, electronic computer, information processing system", "swivel chair", "boat", "bar", "arcade machine", "hovel, hut, hutch, shack, shanty", "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", "towel", "light, light source", "truck, motortruck", "tower", "chandelier, pendant, pendent", "awning, sunshade, sunblind", "streetlight, street lamp", "booth, cubicle, stall, kiosk", "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box", "airplane, aeroplane, plane", "dirt track", "apparel, wearing apparel, dress, clothes", "pole", "land, ground, soil", "bannister, banister, balustrade, balusters, handrail", "escalator, moving staircase, moving stairway", "ottoman, pouf, pouffe, puff, hassock", "bottle", "buffet, counter, sideboard", "poster, posting, placard, notice, bill, card", "stage", "van", "ship", "fountain", "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "canopy", "washer, automatic washer, washing machine", "plaything, toy", "swimming pool, swimming bath, natatorium", "stool", "barrel, cask", "basket, handbasket", "waterfall, falls", "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle", "oven", "ball", "food, solid food", "step, stair", "tank, storage tank", "trade name, brand name, brand, marque", "microwave, microwave oven", "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna", "bicycle, bike, wheel, cycle ", "lake", "dishwasher, dish washer, dishwashing machine", "screen, silver screen, projection screen", "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase", "traffic light, traffic signal, stoplight", "tray", "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin", "fan", "pier, wharf, wharfage, dock", "crt screen", "plate", "monitor, monitoring device", "bulletin board, notice board", "shower", "radiator", "glass, drinking glass", "clock"],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ade()

train_annpath = f'mask2former/datasets/ADE/training.txt'
def ade_train():
    # assert mode in ('train', 'eval', 'test')

    with open(train_annpath, 'r') as fr:
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


def register_ade_train():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']

    name = f"ade_sem_seg_train"
    DatasetCatalog.register(
        name, ade_train
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes=["flag", "wall", "building, edifice", "sky", "floor, flooring", "tree", "ceiling", "road, route", "bed ", "windowpane, window ", "grass", "cabinet", "sidewalk, pavement", "person, individual, someone, somebody, mortal, soul", "earth, ground", "door, double door", "table", "mountain, mount", "plant, flora, plant life", "curtain, drape, drapery, mantle, pall", "chair", "car, auto, automobile, machine, motorcar", "water", "painting, picture", "sofa, couch, lounge", "shelf", "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair", "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press", "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion", "base, pedestal, stand", "box", "column, pillar", "signboard, sign", "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink", "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox", "grandstand, covered stand", "path", "stairs, steps", "runway", "case, display case, showcase, vitrine", "pool table, billiard table, snooker table", "pillow", "screen door, screen", "stairway, staircase", "river", "bridge, span", "bookcase", "blind, screen", "coffee table, cocktail table", "toilet, can, commode, crapper, pot, potty, stool, throne", "flower", "book", "hill", "bench", "countertop", "stove, kitchen stove, range, kitchen range, cooking stove", "palm, palm tree", "kitchen island", "computer, computing machine, computing device, data processor, electronic computer, information processing system", "swivel chair", "boat", "bar", "arcade machine", "hovel, hut, hutch, shack, shanty", "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", "towel", "light, light source", "truck, motortruck", "tower", "chandelier, pendant, pendent", "awning, sunshade, sunblind", "streetlight, street lamp", "booth, cubicle, stall, kiosk", "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box", "airplane, aeroplane, plane", "dirt track", "apparel, wearing apparel, dress, clothes", "pole", "land, ground, soil", "bannister, banister, balustrade, balusters, handrail", "escalator, moving staircase, moving stairway", "ottoman, pouf, pouffe, puff, hassock", "bottle", "buffet, counter, sideboard", "poster, posting, placard, notice, bill, card", "stage", "van", "ship", "fountain", "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "canopy", "washer, automatic washer, washing machine", "plaything, toy", "swimming pool, swimming bath, natatorium", "stool", "barrel, cask", "basket, handbasket", "waterfall, falls", "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle", "oven", "ball", "food, solid food", "step, stair", "tank, storage tank", "trade name, brand name, brand, marque", "microwave, microwave oven", "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna", "bicycle, bike, wheel, cycle ", "lake", "dishwasher, dish washer, dishwashing machine", "screen, silver screen, projection screen", "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase", "traffic light, traffic signal, stoplight", "tray", "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin", "fan", "pier, wharf, wharfage, dock", "crt screen", "plate", "monitor, monitoring device", "bulletin board, notice board", "shower", "radiator", "glass, drinking glass", "clock"],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )

register_ade_train()