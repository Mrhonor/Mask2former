# Automatically find unified label space from co-occurance of detector results.
import sys

sys.path.insert(0, '.')
import os.path as osp
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from collections import defaultdict
from pycocotools import mask as maskutils
import numba
import time
from lib.get_dataloader import get_data_loader
from tqdm import tqdm
import pickle
from lib.models import model_factory
from tools.configer import Configer
from contextlib import ExitStack, contextmanager


from detectron2.engine.hooks import HookBase
import datetime
from detectron2.utils.logger import log_every_n_seconds
# # datapath 
# ROOT_PATH = 'datasets/'

# sys.path.insert(0, ROOT_PATH)
# meta_data_path = ROOT_PATH + 'metadata/det_categories.json' # category info from annotation json
# ANN_PATH = {
#     'objects365': ROOT_PATH + 'objects365/annotations/objects365_val.json',
#     'coco': ROOT_PATH + 'coco/annotations/instances_val2017.json',
#     'oid': ROOT_PATH + 'oid/annotations/oid_challenge_2019_val_expanded.json'
# }

# EXP_NAME = 'Partitioned_COI_RS101_2x'
# DATA_PATH = {
#     'objects365': ROOT_PATH + 'logits/{}/inference_objects365_val/unified_instances_results.json'.format(EXP_NAME),
#     'coco': ROOT_PATH + 'logits/{}/inference_coco_2017_val/unified_instances_results.json'.format(EXP_NAME),
#     'oid': ROOT_PATH + 'logits/{}/inference_oid_val_expanded/unified_instances_results.json'.format(EXP_NAME),
# }
class UniDetLearnUnifyLabelSpace(HookBase):
    @torch.no_grad()
    def after_train(self):
        datasets = ['city', 'mapi', 'sun', 'bdd', 'idd', 'ade', 'coco']
        city_lb = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        mapi_lb = ["Bird", "Ground Animal", "Curb", "Fence", "Guard Rail", "Barrier", "Wall", "Bike Lane", "Crosswalk - Plain", "Curb Cut", "Parking", "Pedestrian Area", "Rail Track", "Road", "Service Lane", "Sidewalk", "Bridge", "Building", "Tunnel", "Person", "Bicyclist", "Motorcyclist", "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General", "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water", "Banner", "Bench", "Bike Rack", "Billboard", "Catch Basin", "CCTV Camera", "Fire Hydrant", "Junction Box", "Manhole", "Phone Booth", "Pothole", "Street Light", "Pole", "Traffic Sign Frame", "Utility Pole", "Traffic Light", "Traffic Sign (Back)", "Traffic Sign (Front)", "Trash Can", "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck", "Wheeled Slow", "Car Mount", "Ego Vehicle"]
        sun_lb = [ "bag", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books", "refridgerator", "television", "paper", "towel", "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub"]
        bdd_lb = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        idd_lb = ["road", "drivable fallback or parking", "sidewalk", "non-drivable fallback or rail track", "person or animal", "out of roi or rider", "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus", "trailer or caravan or vehicle fallback", "curb", "wall", "fence", "guard rail", "billboard", "traffic sign", "traffic light", "polegroup or pole", "obs-str-bar-fallback", "building", "tunnel or bridge", "vegetation", "sky or fallback background"]
        ade_lb = ["flag", "wall", "building, edifice", "sky", "floor, flooring", "tree", "ceiling", "road, route", "bed ", "windowpane, window ", "grass", "cabinet", "sidewalk, pavement", "person, individual, someone, somebody, mortal, soul", "earth, ground", "door, double door", "table", "mountain, mount", "plant, flora, plant life", "curtain, drape, drapery, mantle, pall", "chair", "car, auto, automobile, machine, motorcar", "water", "painting, picture", "sofa, couch, lounge", "shelf", "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair", "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press", "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion", "base, pedestal, stand", "box", "column, pillar", "signboard, sign", "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink", "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox", "grandstand, covered stand", "path", "stairs, steps", "runway", "case, display case, showcase, vitrine", "pool table, billiard table, snooker table", "pillow", "screen door, screen", "stairway, staircase", "river", "bridge, span", "bookcase", "blind, screen", "coffee table, cocktail table", "toilet, can, commode, crapper, pot, potty, stool, throne", "flower", "book", "hill", "bench", "countertop", "stove, kitchen stove, range, kitchen range, cooking stove", "palm, palm tree", "kitchen island", "computer, computing machine, computing device, data processor, electronic computer, information processing system", "swivel chair", "boat", "bar", "arcade machine", "hovel, hut, hutch, shack, shanty", "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", "towel", "light, light source", "truck, motortruck", "tower", "chandelier, pendant, pendent", "awning, sunshade, sunblind", "streetlight, street lamp", "booth, cubicle, stall, kiosk", "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box", "airplane, aeroplane, plane", "dirt track", "apparel, wearing apparel, dress, clothes", "pole", "land, ground, soil", "bannister, banister, balustrade, balusters, handrail", "escalator, moving staircase, moving stairway", "ottoman, pouf, pouffe, puff, hassock", "bottle", "buffet, counter, sideboard", "poster, posting, placard, notice, bill, card", "stage", "van", "ship", "fountain", "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "canopy", "washer, automatic washer, washing machine", "plaything, toy", "swimming pool, swimming bath, natatorium", "stool", "barrel, cask", "basket, handbasket", "waterfall, falls", "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle", "oven", "ball", "food, solid food", "step, stair", "tank, storage tank", "trade name, brand name, brand, marque", "microwave, microwave oven", "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna", "bicycle, bike, wheel, cycle ", "lake", "dishwasher, dish washer, dishwashing machine", "screen, silver screen, projection screen", "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase", "traffic light, traffic signal, stoplight", "tray", "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin", "fan", "pier, wharf, wharfage, dock", "crt screen", "plate", "monitor, monitoring device", "bulletin board, notice board", "shower", "radiator", "glass, drinking glass", "clock"]
        coco_lb = ["rug-merged", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged"]

        # load meta data
        # categories = json.load(open(meta_data_path, 'r'))
        num_cats = [len(eval(d+'_lb')) for d in datasets]
        num_cats_by_name = {}
        for d, n_cat in zip(datasets, num_cats):
            num_cats_by_name[d] = n_cat
        total_cats = sum(num_cats)
        cnt = 0
        dataset_range = {}
        for d, c in zip(datasets, num_cats):
            dataset_range[d] = range(cnt, cnt + c)
        cnt = cnt + c
        print('dataset_range', dataset_range)
        id2source = np.concatenate(
            [np.ones(len(dataset_range[d]), dtype=np.int32) * i \
                for i, d in enumerate(datasets)]
        ).tolist()
        predid2name, id2sourceid, id2sourceindex, id2sourcename = [], [], [], []
        names = []
        for d in datasets:
            predid2name.extend([d + '_' + lb_name for lb_name in eval(d+'_lb')])
            id2sourceid.extend([i for i in range(len(eval(d+'_lb')))])
            id2sourceindex.extend([i for i in range(len(eval(d+'_lb')))])
            id2sourcename.extend([d for _ in range(len(eval(d+'_lb')))])
            names.extend([d + '_' + lb_name for lb_name in eval(d+'_lb')])
            # print('len(categories[d])', d, len(categories[d]))
            
        @torch.no_grad()
        def eval_cross_head_and_datasets(net):
            
            ignore_label = 255
            datasets_cats = self.trainer.cfg.DATASETS.DATASETS_CATS
            n_datasets = len(datasets_cats)
            ignore_index = self.trainer.cfg.DATASETS.IGNORE_LB
            total_cats = 0
            net.aux_mode = 'train'
            net.eval()
            unify_prototype = net.unify_prototype
            # print(unify_prototype.shape)
            bipart_graph = net.bipartite_graphs
            # is_dist = dist.is_initialized()
            dls = get_data_loader(configer, aux_mode='train', distributed=False)
            dataset_id = [0,1,2,3,4,5,6]
            datasets_name = ['city', 'mapi', 'sun', 'bdd', 'idd', 'ade', 'coco']
            for i in dataset_id:
                total_cats += datasets_cats[i]
            # for i in range(0, n_datasets):
            #     total_cats += configer.get("dataset"+str(i+1), "n_cats")
            # total_cats = int(total_cats * configer.get('GNN', 'unify_ratio'))

            heads, mious, target_bipart = [], [], []
            heads.append('single_scale')
            predHist = {}
            for id, i in enumerate(dataset_id):
                this_dataset_cat = datasets_cats[i]
                n_classes = [datasets_cats[j] for j in dataset_id]
                print(n_classes)
                hist = [torch.zeros(this_dataset_cat, n_class) for n_class in n_classes]

                diter = enumerate(tqdm(dls[i]))
                    
                
                with torch.no_grad():
                    for _, (imgs, label) in diter:


                        label = label.squeeze(1).cuda()
                        size = label.shape[-2:]

                        im_sc = imgs.cuda()
                        
                        emb = net(im_sc, dataset=i)
                        # logits = []
                        keep = label != ignore_label
                        cnt = 0
                        cur_id = 0
                        for idx in range(n_datasets):
                            this_cat = configer.get("dataset"+str(idx+1), "n_cats")
                            if idx in dataset_id:
                                # print(this_cat)
                                this_logit = torch.einsum('bchw, nc -> bnhw', emb['seg'], unify_prototype[cnt:cnt+this_cat])
                                this_logit = F.interpolate(this_logit, size=size,
                                    mode='bilinear', align_corners=True)

                                probs = torch.softmax(this_logit, dim=1)
                                preds = torch.argmax(probs, dim=1)

                                hist[cur_id] += torch.tensor(np.bincount(
                                    label.cpu().numpy()[keep.cpu().numpy()] * this_cat + preds.cpu().numpy()[keep.cpu().numpy()],
                                    minlength=this_cat * this_dataset_cat
                                )).view(this_dataset_cat, this_cat)
                                cur_id += 1
                                # logits.append(this_logit)
                            cnt += this_cat
                                
                                

                hist_map = {}
                for idx, name in enumerate(datasets_name):
                    hist_map[name] = hist[idx]
                predHist[datasets_name[id]] = hist_map
            
            return predHist
                        
        if osp.exists("Predhist_7.pickle"):
            with open("Predhist_7.pickle", "rb") as file:
                Predhist = pickle.load(file)
        else:
            def set_model(configer):
                net = model_factory[configer.get('model_name')](configer)

                if configer.get('train', 'finetune'):
                    state = torch.load(configer.get('train', 'finetune_from'), map_location='cpu')
                    
                    if 'model_state_dict' in state:
                        net.load_state_dict(state['model_state_dict'], strict=False)
                    else:
                        net.load_state_dict(state, strict=False)
                        

                    
                if configer.get('use_sync_bn'): 
                    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
                net.cuda()
                net.train()
                return net
            configer = Configer(configs='configs/ltbgnn_7_datasets_snp.json')
            net = set_model(configer)
            Predhist = eval_cross_head_and_datasets(configer, net)
            with open("Predhist_7.pickle", "wb") as file:
                pickle.dump(Predhist, file)

            # 从文件中加载对象
        # for d in datasets:
        #     newPredDist = {}
        #     for idx, name in enumerate(datasets):
        #         newPredDist[name] = Predhist[d][idx]
        #     Predhist[d] = newPredDist

        # with open("Predhist.pickle", "wb") as file:
        #     pickle.dump(Predhist, file)

            
        # function to load detection results
        def create_index(boxes, is_gt=False, score_thresh=-1, cats=None):
            '''
            inputs:
            boxes: list of dicts in coco format {'image_id', 'category_id', 'bbox', 'scores'}
            returns:
            dict of (dict of (list of dict)): per category predictions
            '''
            if cats is not None:
                ret = {x: defaultdict(list) for x in cats}
            else:
                ret = {}
            for x in boxes:
                if x['category_id'] not in ret:
                    ret[x['category_id']] = defaultdict(list)
                if is_gt:
                    ret[x['category_id']][x['image_id']].append(
                        {'bbox': x['bbox'], 'iscrowd': x['iscrowd']})
                else:
                    if x['score'] > score_thresh:
                        ret[x['category_id']][x['image_id']].append(
                            {'bbox': x['bbox'], 'score': x['score']})
            for cat in ret:
                for image_id in ret[cat]:
                    if not is_gt:
                        ret[cat][image_id] = sorted(ret[cat][image_id], key=lambda x: - x['score'])
                        scores = [x['score'] for x in ret[cat][image_id]]
                        bboxes = [x['bbox'] for x in ret[cat][image_id]]
                        ret[cat][image_id] = {'scores': scores, 'bboxes': bboxes}
                    else:
                        ret[cat][image_id] = sorted(ret[cat][image_id], key=lambda x: x['iscrowd'])
                        iscrowd = [x['iscrowd'] for x in ret[cat][image_id]]
                        bboxes = [x['bbox'] for x in ret[cat][image_id]]
                        ret[cat][image_id] = {'iscrowd': iscrowd, 'bboxes': bboxes}
            return ret

        # load detection annotations and predictions
        all_anns, all_preds, gtid2name = {}, {}, {}
        for d in datasets:
            # print('Loading anns...', ANN_PATH[d])
            # anns = json.load(open(ANN_PATH[d], 'r'))
            # print('Loading preds...', DATA_PATH[d])
            # preds = json.load(open(DATA_PATH[d], 'r'))
            # all_anns[d] = create_index(anns['annotations'], True, 
            #                             cats=[x['id'] for x in anns['categories']])
            all_preds[d] = Predhist[d] #create_index(preds, score_thresh=0.1)
            gtid2name[d] = eval(d+'_lb') # {x['id']: x['name'] for x in anns['categories']}
            # del anns
            # del preds
            
        # Predhist: 数据集d : d x n x m维，d为数据集数量， n为原标签预测结果，m为每个标签对应的预测结果
            
        @numba.jit(nopython=True, nogil=True)
        def find_match(ious, iscrowd, n, m):
            matched = [0 for _ in range(m)] # m
            dtig = [0 for _ in range(n)] # n
            dtm = [0 for _ in range(n)] # n
            for dind in range(n):
                iou = 0.5
                mt = -1
                for gind in range(m):
                    if matched[gind] > 0 and iscrowd[gind] == 0:
                        continue
                    if mt > -1 and iscrowd[mt] == 0 and iscrowd[gind] == 1:
                        break
                    if ious[dind, gind] < iou:
                        continue
                    iou = ious[dind, gind]
                    mt = gind
                if mt == -1:
                    continue
                dtig[dind] = iscrowd[mt]
                dtm[dind] = 1
                matched[mt] = 1
            return matched, dtig, dtm

        # COCO mAP
        def numba_mAP(preds, gts):
            '''
            inputs: 
            preds: dict of (list of dict): prediction list indexed by image id (one category)
            gts: dict of (list of dict): prediction list indexed by image id (one category)
            '''
            image_ids = gts.keys()
            all_scores, all_dtm, all_dtig, all_gtig = [], [], [], []
            cur_gt, cur_match = 0, 0
            for idx, image_id in enumerate(image_ids):
                # pruning for efficiency
                if cur_gt > 100 and cur_match < 10:
                    return 0
                if image_id in preds and ('scores' in preds[image_id]):
                    scores = preds[image_id]['scores']
                    dt = preds[image_id]['bboxes']
                else:
                    scores = []
                    dt = []
                if 'iscrowd' in gts[image_id]:
                    gt = gts[image_id]['bboxes']
                    iscrowd = gts[image_id]['iscrowd']
                else:
                    gt = []
                    iscrowd = []
                n, m = len(dt), len(gt)
                ious = maskutils.iou(dt, gt, iscrowd) # n x m
                if n > 0 and m > 0:
                    matched, dtig, dtm = find_match(ious, iscrowd, n, m)
                else:
                    matched = [0 for _ in range(m)] # m
                    dtig = [0 for _ in range(n)] # n
                    dtm = [0 for _ in range(n)] # n
                cur_gt = cur_gt + len(gt)
                cur_match = cur_match + sum(matched)
                all_scores.extend(scores)
                all_dtm.extend(dtm)
                all_dtig.extend(dtig)
                all_gtig.extend(iscrowd)
            all_scores = np.array(all_scores, dtype=np.float32)
            all_dtm = np.array(all_dtm, dtype=np.int32)
            all_dtig = np.array(all_dtig, dtype=np.int32)
            inds = np.argsort(-all_scores, kind='mergesort')
            all_scores = all_scores[inds]
            all_dtm = all_dtm[inds]
            all_dtig = all_dtig[inds]
            N = len(all_scores)
            tps = (all_dtm == 1) & (all_dtig == 0) # N
            fps = (all_dtm == 0) & (all_dtig == 0) # N
            nvalid = np.count_nonzero(np.array(all_gtig, dtype=np.int32)==0)
            if nvalid == 0:
                return 0
            tp_sum = np.cumsum(tps).astype(dtype=np.float32) # N
            fp_sum = np.cumsum(fps).astype(dtype=np.float32) # N
            rc = tp_sum / nvalid
            pr = (tp_sum / (fp_sum + tp_sum + 1e-8)).tolist()
            for i in range(N-1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]
            recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
            inds = np.searchsorted(rc, recThrs, side='left')
            q = np.zeros(len(recThrs))
            try:
                for ri, pi in enumerate(inds):
                    q[ri] = pr[pi]
            except:
                pass
            return q.mean()

        def seg_AP(preds, d, ori_cat, pred_cat):
            return preds[d][ori_cat][pred_cat] / torch.sum(preds[d][ori_cat])
            
        # Pre-calulate mAP for all class pairs
        mAPs_all = {}
        mAPs_ori = {}
        for d in datasets:
            print('d', d)
            mAPs_all[d] = np.zeros((total_cats, num_cats_by_name[d]), dtype=np.float32)
            mAPs_ori[d] = np.zeros(num_cats_by_name[d], dtype=np.float32)
            time_st = time.time()
            for j in range(num_cats_by_name[d]):
                for i in range(total_cats):
                    pred_name = predid2name[i]
                    source_D = id2sourcename[i]
                    source_id = id2sourceindex[i]
                    if source_D == d:
                        continue
                    mAPs_all[d][i, j] = seg_AP(all_preds[d], source_D, j, source_id)
                    
                inds = np.argsort(-mAPs_all[d][:, j])
                # i_ori = [i for i in all_preds[d].keys() if predid2name[i] == '{}_{}'.format(d, gtid2name[d][gt_cat])]
                # i_ori = i_ori[0]
                ori_map = seg_AP(all_preds[d], d, j, j)
                mAPs_all[d][j, j] = ori_map
                mAPs_ori[d][j] = ori_map
                print('{} {:.3f}'.format(gtid2name[d][j], ori_map), end=',')
                for k in range(5):
                    if mAPs_all[d][inds[k], j] > 0.01:
                        print('{} {:.3f}'.format(predid2name[inds[k]], mAPs_all[d][inds[k], j]), end=', ')
                print()
            print('Pre-calulate mAP for all class pairs. time =', time.time() - time_st)
            
        @numba.jit(nopython=True, nogil=True)
        def np_nms(boxes, scores, thresh):
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2] + boxes[:, 0]
            y2 = boxes[:, 3] + boxes[:, 1]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(ovr <= thresh)[0]
                order = order[inds + 1]

            return keep

        def merge_boxes(boxes_list):
            all_image_ids = set()
            for boxes_cat in boxes_list:
                all_image_ids = all_image_ids.union(boxes_cat.keys())
            ret = {}
            for image_id in all_image_ids:
                all_boxes, all_scores = [], []
                for x in boxes_list:
                    if image_id in x and 'bboxes' in x[image_id]:
                        all_boxes.extend(x[image_id]['bboxes'])
                        all_scores.extend(x[image_id]['scores'])
                if len(all_boxes) == 0:
                    continue
                all_boxes = np.array(all_boxes).reshape(-1, 4)
                all_scores = np.array(all_scores).reshape(-1)
                keep = np_nms(all_boxes, all_scores, 0.5)
                all_boxes = all_boxes[keep]
                all_scores = all_scores[keep]
                all_boxes = all_boxes.tolist()
                all_scores = all_scores.tolist()
                ret[image_id] = {'bboxes': all_boxes, 'scores': all_scores}
                
            return ret


        calced_cost = {}
        def calc_cost(cats):
            cats = sorted(cats)
            if tuple(cats) in calced_cost:
                return calced_cost[tuple(cats)]
            num_cats = len(cats)
            sources = [id2sourcename[c] for c in cats]
            ret = 0
            for c, s in zip(cats, sources):
                ann_ind = id2sourceindex[c] 
                ann_cat = id2sourceid[c]
                ori_mAP = mAPs_all[s][c, ann_ind] 
                
                merged_mAP = np.mean([seg_AP(all_preds[s], dataset, ann_ind, id2sourceindex[cc]) for cc, dataset in zip(cats, sources)])
                # merged_mAP = numba_mAP(merged, all_anns[s][ann_cat])
                ret += ori_mAP - merged_mAP

            calced_cost[tuple(cats)] = ret
            nemes = [predid2name[c] for c in cats]
            print(*nemes, ret)
            return ret


        max_new_nodes = {
            2: 1000, 3: 1000, 4: 1500, 5:1000, 6:1000, 7:500
        }
        n2 = max_new_nodes[2]
        tau = 0.2
        oo = 1000

        # Initialize two-node merge
        dataset_dists = {}
        cnt = 0
        for d1, a in enumerate(datasets):
            for d2, b in enumerate(datasets[d1+1:]):
                dist = np.ones(
                    (len(dataset_range[a]), len(dataset_range[b])), dtype=np.float32) * oo
                for i in range(len(dataset_range[a])):
                    for j in range(len(dataset_range[b])):
                        if mAPs_all[a][j+dataset_range[b][0], i] > max(mAPs_ori[a][i] - tau, 0) and \
                            mAPs_all[b][i+dataset_range[a][0], j] > max(mAPs_ori[b][j] - tau, 0):
                            dist[i][j] = calc_cost([i + dataset_range[a][0], j + dataset_range[b][0]])
                            cnt += 1
                dataset_dists[(d1, d2 + d1 + 1)] = dist

        score_thresh = 0.05
        Q = []
        nodes = {}
        valid_two_nodes = {}
        for (i, j) in dataset_dists:
            a, b = datasets[i], datasets[j]
            ra, rb = dataset_range[a], dataset_range[b]
            dist = dataset_dists[(i, j)] # Ni x Nj
            mask = np.any(
                [dist <= np.partition(
                    dist, min(n2, dist.shape[k])-1, axis=k
                ).take([min(n2, dist.shape[k])-1], axis=k) for k in range(2)], axis=0) # Ni x Nj
            mask = mask & (dist <= score_thresh)
            nodes[(i, j)] = list(zip(dist[mask],
                                *[np.array(x)[w] for x, w in zip((ra, rb), np.where(mask))]))
            valid_two_nodes[(i, j)] = set([tuple(sorted((x[1], x[2]))) for x in nodes[(i, j)]])
            Q.append((i, j))
        print('#valid two node merge:', sum(len(v) for k, v in nodes.items()))

        # three-dataset merge
        def remove_duplicate(list1, list2):
            tmp = list1 + list2
            tmp = sorted(tmp, key=lambda x:x[1:])
            ret = [tmp[0]]
            for i in range(1, len(tmp)):
                if tmp[i][1:] != tmp[i-1][1:]:
                    ret.append(tmp[i])
            return tmp


        def get_new_candidates(candidates, ds, new_dataset_id, score_thresh):
            '''
            condidates: list of (score, id1, id2, ...)
            ds: list of existing dataset ids
            '''
            ret = []
            if len(candidates) == 0:
                print('No candidates for', [datasets[d] for d in ds])
                return ret
            n = len(candidates[0]) - 1
            tonew = dataset_range[datasets[new_dataset_id]]
            for item in candidates:
                cost = item[0]
                ids = list(item[1:])
                sources = [id2source[x] for x in ids]
                new_candidates = []
                for new_id in tonew:
                    valid = True
                    for id, source in zip(ids, sources):
                        dataset_pair = tuple(sorted((source, new_dataset_id)))
                        id_pair = tuple(sorted((id, new_id)))
                        if id_pair not in valid_two_nodes[dataset_pair]:
                            valid = False
                            break
                    if valid:
                        new_cost = calc_cost(sorted(ids + [new_id]))
                        new_candidates.append(
                            (new_cost, *tuple(sorted(ids + [new_id]))))
                new_candidates = sorted(new_candidates)[:max_new_nodes[len(ids) + 1]]
                new_candidates = [x for x in new_candidates if x[0] <= score_thresh]
                ret.extend(new_candidates)
                
            return ret

        left = 0
        while left < len(Q):
            ds = Q[left]
            candidates = nodes[ds]
            left = left + 1
            for k, c in enumerate(datasets):
                if k not in ds:
                    new_datasets = tuple(sorted(list(ds) + [k]))
                    new_candidates = get_new_candidates(candidates, ds, k, score_thresh=tau)
                    if new_datasets in nodes and len(nodes[new_datasets]) > 0:
                        nodes[new_datasets] = remove_duplicate(
                            nodes[new_datasets], new_candidates)
                    else:
                        nodes[new_datasets] = new_candidates
                        Q.append(new_datasets)

        top_clusters = [] # formart: (score, id1, id2, ...)
        for ds, candidates in nodes.items():
            top_clusters.extend(candidates)
        # print([len(c)-1 for c in top_clusters])
        # hist([len(c)-1 for c in top_clusters])


        # Running optimization
        from cylp.cy import CyClpSimplex
        from cylp.py.modeling.CyLPModel import CyLPArray, CyLPModel

        cost = CyLPArray([c[0] * (len(c) - 1.) for c in top_clusters])
        size = CyLPArray([len(c)-2. for c in top_clusters])

        m = CyLPModel()
        x = m.addVariable('x', len(top_clusters), isInt=True)

        weight = 0.5 # lambda in the paper

        # add constraints
        m += 0 <= x
        m += x <= 1

        # No overlapping clusters
        cluster_set = [[] for _ in range(total_cats)]
        for i, c in enumerate(top_clusters):
            for a in c[1:]:
                cluster_set[a].append(i)

        for s in cluster_set:
            if len(s):
                w = np.zeros(len(top_clusters))
                w[s] = 1
                m += CyLPArray(w) * x <= 1
                
        # Objective
        m.objective = sum([(cost - weight * size) * x])

        print('C', m.nVars, m.nCons)

        s = CyClpSimplex(m)
        r = s.primal()
        sol_x = s.primalVariableSolution['x']
        print(r, np.sum(sol_x), total_cats - sum(size * sol_x), sum(cost * sol_x), weight * (1.0 * total_cats - 1.0 * sum(size * sol_x)))
        print('num_classes', int(total_cats - sum(size * sol_x) + 1e-7))


        # Visualize merged classes
        for i, s in enumerate(sol_x):
            if s > 0.1:
                print('%4.1f %7.4f'%(s, top_clusters[i][0]), *['%-50s'%names[j] for j in top_clusters[i][1:]])


        # Print google sheet friendly version
        change_oid_names = ['Mouse', 'Bench']
        # for change_oid_name in change_oid_names:
        #     cnt = 0
        #     for x in categories['oid']:
        #         if x['name'] == change_oid_name:
        #             cnt = cnt + 1
        #             x['name'] = '{}{}'.format(change_oid_name, cnt)
        #             # print('Renaming oid', change_oid_name, \
        #             #       'to', x['name'])
        # oidname2freebase = {x['name']: x['freebase_id'] for x in categories['oid']}
        # oidname2freebase[''] = ''
        names = []
        for d in datasets:
            names.extend(eval(d+'_lb'))
        merged = [False for _ in range(len(names))]
        print_order = datasets
        heads = ['city', 'mapi', 'sun', 'bdd', 'idd', 'ade', 'coco']
        head_str = 'key'
        for head in heads:
            head_str = head_str + ', {}'.format(head)
        print(head_str)
        cnt = 0
        for i, s in enumerate(sol_x):
            if s > 0.1:
                inds = top_clusters[i][1:]
                dataset_name = {d: '' for d in datasets}
                for ind in inds:
                merged[ind] = True
                d = datasets[id2source[ind]]
                name = names[ind]
                dataset_name[d] = name
                # if name == 'background':
                #   continue
                unified_name = dataset_name[print_order[0]].replace(',', '_')
                for d in print_order[1:]:
                    unified_name = unified_name + '_{}'.format(dataset_name[d].replace(',', '_'))
                print(unified_name, end='')
                cnt = cnt + 1
                for d in print_order:
                    # if d == 'oid':
                    #     print(', {}, {}'.format(oidname2freebase[dataset_name[d]], dataset_name[d]), end='')
                    # else:
                    # print(', {}'.format(dataset_name[d]), end='')
                    # print("!:", dataset_name[d])
                    if dataset_name[d] != '':
                        print(', {}'.format(eval(d+'_lb').index(dataset_name[d])), end='')
                    else:
                        print(', {}'.format(dataset_name[d]), end='')
                print()
        for ind in range(len(names)):
        if not merged[ind]:
            dataset_name = {d: '' for d in datasets}
            d = datasets[id2source[ind]]
            name = names[ind]
            # if name == 'background':
            #   continue
            dataset_name[d] = name
            unified_name = dataset_name[print_order[0]].replace(',', '_')
            for d in print_order[1:]:
                unified_name = unified_name + '_{}'.format(dataset_name[d].replace(',', '_'))
            print(unified_name, end='')
            cnt = cnt + 1
            for d in print_order:
                # if d == 'oid':
                #     print(', {}, {}'.format(oidname2freebase[dataset_name[d]], dataset_name[d]), end='')
                # else:
                # print(', {}'.format(dataset_name[d]), end='')
                if dataset_name[d] != '':
                    print(', {}'.format(eval(d+'_lb').index(dataset_name[d])), end='')
                else:
                    print(', {}'.format(dataset_name[d]), end='')
            print()
        print()
        print('num_cats', cnt)

        # cnt = 0
        # for i, s in enumerate(sol_x):
        #     if s > 0.1:
        #         inds = top_clusters[i][1:]
        #         dataset_name = {d: '' for d in datasets}
        #         for ind in inds:
        #           merged[ind] = True
        #           d = datasets[id2source[ind]]
        #           name = names[ind]
        #           dataset_name[d] = name
        #         # if name == 'background':
        #         #   continue
        #         unified_name = dataset_name[print_order[0]]
        #         for d in print_order[1:]:
        #             unified_name = unified_name + '_{}'.format(dataset_name[d])
        #         print(unified_name, end='')
        #         cnt = cnt + 1
        #         for d in print_order:
        #             # if d == 'oid':
        #             #     print(', {}, {}'.format(oidname2freebase[dataset_name[d]], dataset_name[d]), end='')
        #             # else:
        #             print(', {}'.format(dataset_name[d]), end='')
        #         print()
        # for ind in range(len(names)):
        #   if not merged[ind]:
        #     dataset_name = {d: '' for d in datasets}
        #     d = datasets[id2source[ind]]
        #     name = names[ind]
        #     # if name == 'background':
        #     #   continue
        #     dataset_name[d] = name
        #     unified_name = dataset_name[print_order[0]]
        #     for d in print_order[1:]:
        #         unified_name = unified_name + '_{}'.format(dataset_name[d])
        #     print(unified_name, end='')
        #     cnt = cnt + 1
        #     for d in print_order:
        #         # if d == 'oid':
        #         #     print(', {}, {}'.format(oidname2freebase[dataset_name[d]], dataset_name[d]), end='')
        #         # else:
        #         print(', {}'.format(dataset_name[d]), end='')
        #     print()
        # print()
        # print('num_cats', cnt)

