#!/usr/bin/python
# -*- encoding: utf-8 -*-


from email.policy import strict
import sys

sys.path.insert(0, '.')
import os
import os.path as osp


import logging
import argparse
import math
from tabulate import tabulate

from tqdm import tqdm
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import ExitStack, contextmanager

from lib.models import model_factory
from lib.logger import setup_logger
from lib.get_dataloader import get_data_loader, get_city_loader
from tools.configer import Configer
from detectron2.engine.hooks import HookBase
import datetime
from detectron2.utils.logger import log_every_n_seconds

CITY_ID = 0
CAM_ID = 1
A2D2_ID = 2



def get_round_size(size, divisor=32):
    return [math.ceil(el / divisor) * divisor for el in size]

class MscEvalV0(object):

    def __init__(self, scales=(0.5, ), flip=False, ignore_label=255):
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label

    def __call__(self, net, dl, n_classes, dataset_id):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            probs = torch.zeros(
                    (N, n_classes, H, W), dtype=torch.float32).cuda().detach()

            for scale in self.scales:
                sH, sW = int(scale * H), int(scale * W)
                sH, sW = get_round_size((sH, sW))
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                
                logits = net(im_sc, dataset=dataset_id)[0]
                logits = F.interpolate(logits, size=size,
                        mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc, dataset=dataset_id)[0]
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            

            keep = label != self.ignore_label
            hist += torch.tensor(np.bincount(
                label.cpu().numpy()[keep.cpu().numpy()] * n_classes + preds.cpu().numpy()[keep.cpu().numpy()],
                minlength=n_classes ** 2
            )).cuda().view(n_classes, n_classes)
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        print(ious)
        miou = np.nanmean(ious.detach().cpu().numpy())
        return miou.item()

class MscEvalV0_Contrast(object):

    def __init__(self, configer, scales=(0.5, ), flip=False, ignore_label=255, ori_scales=False):
        self.configer = configer
        # self.num_unify_classes = self.configer.get('num_unify_classes')
        # self.class_Remaper = ClassRemap(configer=self.configer)
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label
        self.ori_scales = ori_scales

        # self.lb_map = torch.tensor(np.load('mapi_relabel.npy')).cuda()
        # print(self.lb_map)

    def __call__(self, net, dl, n_classes, dataset_id):
        # n_classes = 43
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        # hist = torch.zeros(118, 118).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            # probs = torch.zeros(
            #         (N, self.num_unify_classes, H, W), dtype=torch.float32).cuda().detach()
            probs = None
            # probs = torch.zeros(
            #         (N, 150, H, W), dtype=torch.float32).cuda().detach()


            for scale in self.scales:

                
                sH, sW = int(scale * H), int(scale * W)
                sH, sW = get_round_size((sH, sW))
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                
                logits = net(im_sc, dataset=dataset_id)
                # logits = net(im_sc, dataset_id * torch.ones(N, dtype=torch.long))
                N, _, lH, lW = logits.shape
                if self.ori_scales:
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    if probs is None:
                        probs = torch.zeros(
                            (N, n_classes, H, W), dtype=torch.float32).cuda().detach()
                else:
                    label = F.interpolate(label.float().unsqueeze(1), size=(lH, lW),
                            mode='nearest').squeeze(1).long()
                    if probs is None:
                        probs = torch.zeros(
                            (N, n_classes, lH, lW), dtype=torch.float32).cuda().detach()
                    # label = F.interpolate()
                

                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc, dataset=dataset_id)
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            keep = label != self.ignore_label
            
            # print(np.max(label.cpu().numpy()[keep.cpu().numpy()]))

            hist += torch.tensor(np.bincount(
                label.cpu().numpy()[keep.cpu().numpy()] * n_classes + preds.cpu().numpy()[keep.cpu().numpy()],
                minlength=n_classes ** 2
            )).cuda().view(n_classes, n_classes)
            # hist += torch.tensor(np.bincount(
            #     label.cpu().numpy()[keep.cpu().numpy()] * 118 + preds.cpu().numpy()[keep.cpu().numpy()],
            #     minlength=118 ** 2
            # )).cuda().view(118, 118)
                
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        print(ious)
        miou = np.nanmean(ious.detach().cpu().numpy())
        return miou.item()

class MscEvalV0_unlabel(object):

    def __init__(self, configer, scales=(0.5, ), flip=False, ignore_label=255, ori_scales=False):
        self.configer = configer
        # self.num_unify_classes = self.configer.get('num_unify_classes')
        # self.class_Remaper = ClassRemap(configer=self.configer)
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label
        self.ori_scales = ori_scales

        # self.lb_map = torch.tensor(np.load('mapi_relabel.npy')).cuda()
        # print(self.lb_map)

    def __call__(self, net, dl, n_classes, dataset_id):
        # n_classes = 43
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        # hist = torch.zeros(118, 118).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            # probs = torch.zeros(
            #         (N, self.num_unify_classes, H, W), dtype=torch.float32).cuda().detach()
            probs = None
            # probs = torch.zeros(
            #         (N, 150, H, W), dtype=torch.float32).cuda().detach()


            for scale in self.scales:

                
                sH, sW = int(scale * H), int(scale * W)
                sH, sW = get_round_size((sH, sW))
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                
                ori_logits = net(im_sc, dataset=dataset_id)
                # logits = net(im_sc, dataset_id * torch.ones(N, dtype=torch.long))
                N, D, lH, lW = ori_logits.shape
                logits = ori_logits[:, :n_classes, :, :]
                if self.ori_scales:
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    if probs is None:
                        probs = torch.zeros(
                            (N, n_classes, H, W), dtype=torch.float32).cuda().detach()
                else:
                    label = F.interpolate(label.float().unsqueeze(1), size=(lH, lW),
                            mode='nearest').squeeze(1).long()
                    if probs is None:
                        probs = torch.zeros(
                            (N, n_classes, lH, lW), dtype=torch.float32).cuda().detach()
                    # label = F.interpolate()
                

                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc, dataset=dataset_id)
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            keep = label != self.ignore_label
            
            # print(np.max(label.cpu().numpy()[keep.cpu().numpy()]))

            hist += torch.tensor(np.bincount(
                label.cpu().numpy()[keep.cpu().numpy()] * n_classes + preds.cpu().numpy()[keep.cpu().numpy()],
                minlength=n_classes ** 2
            )).cuda().view(n_classes, n_classes)
            # hist += torch.tensor(np.bincount(
            #     label.cpu().numpy()[keep.cpu().numpy()] * 118 + preds.cpu().numpy()[keep.cpu().numpy()],
            #     minlength=118 ** 2
            # )).cuda().view(118, 118)
                
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        print(ious)
        miou = np.nanmean(ious.detach().cpu().numpy())
        return miou.item()

class MscEvalV0_AutoLink(object):

    def __init__(self, configer, scales=(0.5, ), flip=False, ignore_label=255):
        self.configer = configer
        self.n_datasets = self.configer.get('n_datasets')
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label

    def __call__(self, net, dl, n_classes, dataset_id):
        ## evaluate
        # hist = torch.zeros(n_classes, n_classes).cuda().detach()
        datasets_remap = []
        # hist = torch.zeros(n_classes, n_classes).cuda().detach()
        # if dist.is_initialized() and dist.get_rank() != 0:
        diter = enumerate(dl)
        # else:
        #     diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            # print(size)

            scale = self.scales[0]
            sH, sW = int(scale * H), int(scale * W)
            sH, sW = get_round_size((sH, sW))
            im_sc = F.interpolate(imgs, size=(sH, sW),
                    mode='bilinear', align_corners=True)

            im_sc = im_sc.cuda()
            all_logits = net(im_sc)
            for index in range(0, self.n_datasets):
                if index == dataset_id:
                    if len(datasets_remap) <= index:
                        datasets_remap.append(torch.eye(n_classes).cuda().detach())
                     
                    continue
                
                n_cats = self.configer.get('dataset'+str(index+1), 'n_cats')
                this_data_hist = torch.zeros(n_classes, n_cats).cuda().detach()

                logits = F.interpolate(all_logits[index], size=size,
                        mode='bilinear', align_corners=True)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                keep = label != self.ignore_label
        
                this_data_hist = torch.tensor(np.bincount(
                    label.cpu().numpy()[keep.cpu().numpy()] * n_cats + preds.cpu().numpy()[keep.cpu().numpy()],
                    minlength=n_classes * n_cats
                )).cuda().view(n_classes, n_cats)
                if len(datasets_remap) <= index:
                    datasets_remap.append(this_data_hist)
                else:
                    datasets_remap[index] += this_data_hist
                         
        return [torch.argmax(hist, dim=1) for hist in datasets_remap]
        # if dist.is_initialized():
        #     dist.all_reduce(hist, dist.ReduceOp.SUM)
            
        # ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        # print(ious)
        # miou = np.nanmean(ious.detach().cpu().numpy())
        # return miou.item()


# 修改后用于多数据集
def eval_model_label_link(configer, net):
    org_aux = net.aux_mode
    net.aux_mode = 'eval'

    is_dist = dist.is_initialized()

    n_datasets = configer.get("n_datasets")

    dls = get_data_loader(configer, aux_mode='eval', distributed=is_dist)

    net.eval()

    heads, mious = [], []
    logger = logging.getLogger()

    single_scale = MscEvalV0_Contrast(configer, (1., ), False)
    
    for i in range(0, configer.get('n_datasets')):
        mIOU = single_scale(net, dls[i], configer.get('dataset'+str(i+1),"n_cats"), i)
        mious.append(mIOU)
    
    heads.append('single_scale')
    # mious.append(mIOU_cam)
    # mious.append(mIOU_city)
    # mious.append(mIOU_a2d2)
    # logger.info('Cam single mIOU is: %s\nCityScapes single mIOU is: %s\n A2D2 single mIOU is: %s\n', mIOU_cam, mIOU_city, mIOU_a2d2)
    # logger.info('Cam single mIOU is: %s\nCityScapes single mIOU is: %s\n', mIOU_cam, mIOU_city)

    net.aux_mode = org_aux
    return heads, mious

def main():
    # 修改后用于多数据集
    args = parse_args()
    configer = Configer(configs=args.config)


    # if not args.local_rank == -1:
    #     torch.cuda.set_device(args.local_rank)
    #     dist.init_process_group(backend='nccl',
    #     init_method='tcp://127.0.0.1:{}'.format(args.port),
    #     world_size=torch.cuda.device_count(),
    #     rank=args.local_rank
    # )
    if not osp.exists(configer.get('res_save_pth')): os.makedirs(configer.get('res_save_pth'))
    setup_logger('{}-eval'.format(configer.get('model_name')), configer.get('res_save_pth'))
    
    logger = logging.getLogger()
    net = model_factory[configer.get('model_name')](configer)
    state = torch.load('res/celoss/train3_seg_model_stage1_150000.pth', map_location='cpu')
    net.load_state_dict(state, strict=False)
    
    net.cuda()
    net.aux_mode = 'eval'
    net.eval()
    
    # graph_net = model_factory[configer.get('GNN','model_name')](configer)
    # torch.set_printoptions(profile="full")
    # graph_net.load_state_dict(torch.load('res/celoss/graph_model_270000.pth', map_location='cpu'), strict=False)
    # graph_net.cuda()
    # graph_net.eval()
    # # graph_node_features = gen_graph_node_feature(configer)
    # graph_node_features = torch.load('res/celoss/graph_node_features5_CityScapes_CamVid_Sunrgbd_Bdd100k_Idd.pt')
    # # unify_prototype, ori_bi_graphs = graph_net.get_optimal_matching(graph_node_features, init=True) 
    # # unify_prototype, ori_bi_graphs,_,_ = graph_net(graph_node_features)
    # unify_prototype, ori_bi_graphs = graph_net.get_optimal_matching(graph_node_features, init=True) 
    # bi_graphs = []
    # if len(ori_bi_graphs) == 10:
    #     for j in range(0, len(ori_bi_graphs), 2):
    #         bi_graphs.append(ori_bi_graphs[j+1].detach())
    # else:
    #     bi_graphs = [bigh.detach() for bigh in ori_bi_graphs]
    # # unify_prototype, bi_graphs, adv_out, _ = graph_net(graph_node_features)

    # # print(bi_graphs[0])
    # # print(bi_graphs[0][18])
    # print(torch.norm(net.unify_prototype[0][0], p=2))
    # print(torch.norm(unify_prototype[0][0], p=2))
    # net.set_unify_prototype(unify_prototype)
    # net.set_bipartite_graphs(bi_graphs) 
    
    heads, mious = eval_model_contrast(configer, net)
    
    # heads, mious = eval_model_emb(configer, net)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
    
    
def Find_label_relation(configer, datasets_remaps):
    n_datasets = configer.get('n_datasets')
    out_label_relation = []
    total_cats = 0
    dataset_cats = []
    for i in range(0, n_datasets):
        dataset_cats.append(configer.get('dataset'+str(i+1), 'n_cats'))
        total_cats += configer.get('dataset'+str(i+1), 'n_cats')
        
    bipart_graph =torch.zeros((total_cats, total_cats), dtype=torch.float) 
    for i in range(0, n_datasets):
        this_datasets_sets = datasets_remaps[i]
        for j in range(i+1, n_datasets):

            this_datasets_map = datasets_remaps[i][j]
            other_datasets_map = datasets_remaps[j][i]
            this_size = len(this_datasets_map)+len(other_datasets_map)
            this_label_relation = torch.zeros((this_size, this_size), dtype=torch.bool)
            
            for index, val in enumerate(this_datasets_map):
                this_label_relation[index][len(this_datasets_map)+val] = True
            
            for index, val in enumerate(other_datasets_map):
                this_label_relation[len(this_datasets_map)+index][val] = True
            out_label_relation.append(this_label_relation)
        
    return out_label_relation
    # conflict = []
    
@torch.no_grad()
def find_unuse_label(configer, net, dl, n_classes, dataset_id):
        ## evaluate
    # hist = torch.zeros(n_classes, n_classes).cuda().detach()
    # datasets_remap = []
    ignore_label = 255
    n_datasets = configer.get("n_datasets")
    total_cats = 0
    net.aux_mode = 'train'
    net.eval()
    unify_prototype = net.unify_prototype
    # print(unify_prototype.shape)
    bipart_graph = net.bipartite_graphs
    
    for i in range(0, n_datasets):
        total_cats += configer.get("dataset"+str(i+1), "n_cats")
    total_cats = int(total_cats * configer.get('GNN', 'unify_ratio'))

    hist = torch.zeros(n_classes, total_cats).cuda().detach()
    if dist.is_initialized() and dist.get_rank() != 0:
        diter = enumerate(dl)
    else:
        diter = enumerate(tqdm(dl))
        
    
    with torch.no_grad():
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape
            if H > 2048 or W > 2048:
                H = 2048
                W = 2048
                

            label = label.squeeze(1).cuda()
            size = label.shape[-2:]

            im_sc = F.interpolate(imgs, size=(H, W),
                    mode='bilinear', align_corners=True)

            im_sc = im_sc.cuda()
            
            emb = net(im_sc, dataset=dataset_id)
        
            logits = torch.einsum('bchw, nc -> bnhw', emb['seg'], unify_prototype)

            logits = F.interpolate(logits, size=size,
                    mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != ignore_label

            hist += torch.tensor(np.bincount(
                label.cpu().numpy()[keep.cpu().numpy()] * total_cats + preds.cpu().numpy()[keep.cpu().numpy()],
                minlength=n_classes * total_cats
            )).cuda().view(n_classes, total_cats)

    max_value, max_index = torch.max(bipart_graph[dataset_id], dim=0)
    # print(max_value)
    n_cat = configer.get(f'dataset{dataset_id+1}', 'n_cats')
    
    # torch.set_printoptions(profile="full")
    # print(hist)

    buckets = {}
    for index, j in enumerate(max_index):
        if max_value[index] == 0:
            continue
        
        if int(j) not in buckets:
            buckets[int(j)] = [index]
        else:
            buckets[int(j)].append(index)

    for index in range(0, n_cat):
        if index not in buckets:
            print('index not in buckets:', index)
            buckets[index] = []

    for index, val in buckets.items():
        total_num = 0
        for i in val:
            total_num += hist[index][i]
        new_val = []
        if total_num != 0:
            for i in val:
                rate = hist[index][i] / total_num
                if rate > 1e-5:
                    new_val.append(i)
        
        buckets[index] = new_val

    net.train()
    return buckets 


@torch.no_grad()
def eval_find_use_and_unuse_label(n_datasets, net):
        ## evaluate
    # hist = torch.zeros(n_classes, n_classes).cuda().detach()
    # datasets_remap = []
    org_aux = net.aux_mode
    
    ignore_label = 255
    n_datasets = configer.get("n_datasets")
    ignore_index = configer.get('loss','ignore_index')
    total_cats = 0
    net.aux_mode = 'train'
    net.eval()
    unify_prototype = net.unify_prototype
    # print(unify_prototype.shape)
    bipart_graph = net.bipartite_graphs
    is_dist = dist.is_initialized()
    dls = get_data_loader(configer, aux_mode='train', distributed=is_dist, stage=2)
    
    for i in range(0, n_datasets):
        total_cats += configer.get("dataset"+str(i+1), "n_cats")
    total_cats = int(total_cats * configer.get('GNN', 'unify_ratio'))

    heads, mious, target_bipart = [], [], []
    heads.append('single_scale')

    for i in range(0, n_datasets):
        n_classes = configer.get(f'dataset{i+1}', 'n_cats')
        hist = torch.zeros(n_classes, total_cats).cuda().detach()
        # hist_origin = torch.zeros(n_classes, n_classes).cuda().detach()
        
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dls[i])
        else:
            diter = enumerate(tqdm(dls[i]))
            
        
        with torch.no_grad():
            for _, (imgs, label) in diter:
                # N, _, H, W = label.shape
                # if H > 512 or W > 512:
                #     H = 512
                #     W = 512
                    

                # label = label.squeeze(1).cuda()
                # size = label.shape[-2:]

                # im_sc = F.interpolate(imgs, size=(H, W),
                #         mode='bilinear', align_corners=True)

                # im_sc = im_sc.cuda()
            
                N, _, H, W = label.shape


                label = label.squeeze(1).cuda()
                size = label.shape[-2:]

                im_sc = imgs.cuda()
                
                emb = net(im_sc, dataset=i)
            
                logits = torch.einsum('bchw, nc -> bnhw', emb['seg'], unify_prototype)
                # remap_logits = torch.einsum('bchw, nc -> bnhw', logits, bipart_graph[i])

                logits = F.interpolate(logits, size=size,
                        mode='bilinear', align_corners=True)
                # remap_logits = F.interpolate(remap_logits, size=size,
                #         mode='bilinear', align_corners=True)

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # remap_probs = torch.softmax(remap_logits, dim=1)
                # remap_preds = torch.argmax(remap_probs, dim=1)
                
                keep = label != ignore_label

                hist += torch.tensor(np.bincount(
                    label.cpu().numpy()[keep.cpu().numpy()] * total_cats + preds.cpu().numpy()[keep.cpu().numpy()],
                    minlength=n_classes * total_cats
                )).cuda().view(n_classes, total_cats)

                # hist_origin += torch.tensor(np.bincount(
                #     label.cpu().numpy()[keep.cpu().numpy()] * n_classes + remap_preds.cpu().numpy()[keep.cpu().numpy()],
                #     minlength=n_classes ** 2
                # )).cuda().view(n_classes, n_classes)


        # if dist.is_initialized():
        #     dist.all_reduce(hist_origin, dist.ReduceOp.SUM)
        # ious = hist_origin.diag() / (hist_origin.sum(dim=0) + hist_origin.sum(dim=1) - hist_origin.diag())
        # print(ious)
        # miou = np.nanmean(ious.detach().cpu().numpy()).item()
        # mious.append(miou)

        max_value, max_index = torch.max(bipart_graph[i], dim=0)
        # print(max_value)
        n_cat = configer.get(f'dataset{i+1}', 'n_cats')
        
        # torch.set_printoptions(profile="full")
        # print(hist)

        bipart = ignore_index * torch.ones_like(bipart_graph[i])
        buckets = {}
        for index, j in enumerate(max_index):
            if max_value[index] == 0:
                continue
            
            if int(j) not in buckets:
                buckets[int(j)] = [index]
            else:
                buckets[int(j)].append(index)

        for index in range(0, n_cat):
            if index not in buckets:
                print('index not in buckets:', index)
                buckets[index] = []

        for index, val in buckets.items():
            total_num = 0
            sum_num = torch.sum(hist[index])
            for idx in val:
                total_num += hist[index][idx]
            # remove_val = []
            # affirm_val = []
            if total_num != 0:
                for idx in val:
                    rate = hist[index][idx] / total_num
                    if rate < 1e-2:
                        bipart[index][idx] = 0
                        # remove_val.append(i)
                    elif rate > 0.5:
                        bipart[index][idx] = 1
                        # affirm_val.append(i)
            
            # buckets[index] = new_val
        target_bipart.append(bipart)

    # net.train()
    net.aux_mode = org_aux
    return heads, mious, target_bipart 

if __name__ == "__main__":
    main()
    # args = parse_args()
    # configer = Configer(configs=args.config)
    # datasets_remaps = []
    # set0 = []
    # set0.append([])
    # set0.append([2,0])
    # datasets_remaps.append(set0)

    # set1 = []
    # set1.append([0,1,0])
    # set1.append([])
    # datasets_remaps.append(set1)
    # print(Find_label_relation(configer, datasets_remaps))

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

class eval_link_hook(HookBase):
    def after_step(self):
        if self.trainer.iter % 5000 == 0 and self.trainer.model.train_seg_or_gnn==self.trainer.model.GNN:
            logger = logging.getLogger(__name__)
            logger.info(f"eval link at iteration {self.trainer.iter}!")
            # org_aux = net.aux_mode

            model = self.trainer.model
            bipart_graph = model.get_bipart_graph()
            ignore_label = 255
            datasets_cats = self.trainer.cfg.DATASETS.DATASETS_CATS
            n_datasets = len(datasets_cats)
            ignore_index = self.trainer.cfg.DATASETS.IGNORE_LB
            total_cats = 0
            callbacks = None
            for i in range(0, n_datasets):
                total_cats += datasets_cats[i]
            num_unfiy_class = self.trainer.cfg.DATASETS.NUM_UNIFY_CLASS
            heads, mious, target_bipart = [], [], []

            is_dist = dist.is_initialized()
            for idx, dataset_name in enumerate(self.trainer.cfg.DATASETS.EVAL):
                data_loader = self.trainer.cls.build_test_loader(self.trainer.cfg, dataset_name)
        
                n_classes = datasets_cats[idx]
                hist = torch.zeros(n_classes, num_unfiy_class).cuda()
                # hist_origin = torch.zeros(n_classes, n_classes).cuda().detach()    
                
                with torch.no_grad():
                    total = len(data_loader)
                    num_warmup = min(5, total - 1)
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0
                    with ExitStack() as stack:
                        if isinstance(model, nn.Module):
                            stack.enter_context(inference_context(model))
                        stack.enter_context(torch.no_grad())

                        start_data_time = time.perf_counter()
                        dict.get(callbacks or {}, "on_start", lambda: None)()
                        for idx, inputs in enumerate(data_loader):
                            total_data_time += time.perf_counter() - start_data_time
                            if idx == num_warmup:
                                start_time = time.perf_counter()
                                total_data_time = 0
                                total_compute_time = 0
                                total_eval_time = 0

                            start_compute_time = time.perf_counter()
                            dict.get(callbacks or {}, "before_inference", lambda: None)()
                            outputs = model(inputs)
                            dict.get(callbacks or {}, "after_inference", lambda: None)()
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            total_compute_time += time.perf_counter() - start_compute_time

                            start_eval_time = time.perf_counter()
                            labels = [x["sem_seg"].cuda() for x in inputs]
                            logits = outputs["uni_logits"]
                            for lb, lg in zip(labels, logits):
                                lg = F.interpolate(lg, size=lb.size[-2:],
                                        mode='bilinear', align_corners=True)

                                probs = torch.softmax(lg, dim=1)
                                preds = torch.argmax(probs, dim=1)
                                                   
                                keep = lb != ignore_label

                                hist += torch.tensor(np.bincount(
                                    lb.cpu().numpy()[keep.cpu().numpy()] * num_unfiy_class + preds.cpu().numpy()[keep.cpu().numpy()],
                                    minlength=n_classes * num_unfiy_class
                                )).cuda().view(n_classes, num_unfiy_class) 
                            total_eval_time += time.perf_counter() - start_eval_time

                            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                            data_seconds_per_iter = total_data_time / iters_after_start
                            compute_seconds_per_iter = total_compute_time / iters_after_start
                            eval_seconds_per_iter = total_eval_time / iters_after_start
                            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                                log_every_n_seconds(
                                    logging.INFO,
                                    (
                                        f"Inference done {idx + 1}/{total}. "
                                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                        f"ETA={eta}"
                                    ),
                                    n=5,
                                )
                            start_data_time = time.perf_counter()
                        dict.get(callbacks or {}, "on_end", lambda: None)()

                    # Measure the time only for this worker (before the synchronization barrier)
                    total_time = time.perf_counter() - start_time
                    total_time_str = str(datetime.timedelta(seconds=total_time))
                    # NOTE this format is parsed by grep
                    logger.info(
                        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                            total_time_str, total_time / (total - num_warmup), num_devices
                        )
                    )
                    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
                    logger.info(
                        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
                        )
                    )                    
                     

                max_value, max_index = torch.max(bipart_graph[i], dim=0)
                
                # torch.set_printoptions(profile="full")
                # print(hist)

                bipart = ignore_index * torch.ones_like(bipart_graph[i])
                buckets = {}
                for index, j in enumerate(max_index):
                    if max_value[index] == 0:
                        continue
                    
                    if int(j) not in buckets:
                        buckets[int(j)] = [index]
                    else:
                        buckets[int(j)].append(index)

                for index in range(0, n_classes):
                    if index not in buckets:
                        print('index not in buckets:', index)
                        buckets[index] = []

                for index, val in buckets.items():
                    total_num = 0
                    sum_num = torch.sum(hist[index])
                    for idx in val:
                        total_num += hist[index][idx]
                    # remove_val = []
                    # affirm_val = []
                    if total_num != 0:
                        for idx in val:
                            rate = hist[index][idx] / total_num
                            if rate < 1e-2:
                                bipart[index][idx] = 0
                                # remove_val.append(i)
                            elif rate > 0.5:
                                bipart[index][idx] = 1
                                # affirm_val.append(i)
                    
                    # buckets[index] = new_val
                target_bipart.append(bipart)

            # net.train()
            # target_bipart.cat(target_bipart, dim=0)
            model.set_target_bipart(target_bipart)
            