import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from .modeling.transformer_decoder.GNN.gen_graph_node_feature import gen_graph_node_feature
from .modeling.transformer_decoder.GNN.ltbgnn import build_GNN_module
from .modeling.backbone.hrnet_backbone import HighResolutionNet
from .modeling.loss.ohem_ce_loss import OhemCELoss
from timm.models.layers import trunc_normal_
import clip
import logging
from detectron2.utils.events import get_event_storage, EventStorage
import numpy as np

logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class HRNet_W48_ARCH(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """
    @configurable
    def __init__(self, *, 
                backbone,
                gnn_model,
                sem_seg_head,
                datasets_cats,
                with_datasets_aux,
                ignore_lb,
                ohem_thresh,
                size_divisibility,
                pixel_mean,
                pixel_std,
                graph_node_features,
                init_gnn_iters,
                Pretraining,
                gnn_iters,
                seg_iters,
                first_stage_gnn_iters,
                num_unify_classes,
                with_spa_loss
                ):
        super(HRNet_W48_ARCH, self).__init__()
        self.num_unify_classes = num_unify_classes

        self.datasets_cats = datasets_cats
        self.n_datasets = len(self.datasets_cats)
        self.backbone = backbone
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("alter_iters", torch.zeros(1), False)
        self.register_buffer("train_seg_or_gnn", torch.zeros(1), False)
        self.GNN = 1
        self.SEG = 0
        self.init_gnn_iters = init_gnn_iters
        self.Pretraining = Pretraining
        
        self.gnn_iters = gnn_iters
        self.seg_iters = seg_iters
        self.first_stage_gnn_iters = first_stage_gnn_iters
        self.sec_stage_gnn_iters = gnn_iters - first_stage_gnn_iters
        self.with_datasets_aux = with_datasets_aux
        assert self.first_stage_gnn_iters < self.gnn_iters, "first_stage_gnn_iters must less than gnn_iters"
        self.proj_head = sem_seg_head # ProjectionHead(dim_in=in_channels, proj_dim=self.output_feat_dim, bn_type=bn_type)
        self.graph_node_features = graph_node_features.cuda()
        self.total_cats = 0
        # self.datasets_cats = []
        for i in range(0, self.n_datasets):
            # self.datasets_cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
            self.total_cats += self.datasets_cats[i]
 
        self.criterion = OhemCELoss(ohem_thresh, ignore_lb)
        
        # 初始化 grad
        self.initial = False
        self.inFirstGNNStage = True
 
        #  if self.MODEL_WEIGHTS != None:
        # state = torch.load('output/pretrain_model_30000.pth')
        # self.load_state_dict(state['model_state_dict'], strict=True)
        self.isLoad = False
        self.with_spa_loss = with_spa_loss

        # self.get_encode_lb_vec()

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, 720)
        gnn_model = None #build_GNN_module(cfg)
        datasets_cats = cfg.DATASETS.DATASETS_CATS
        ignore_lb = cfg.DATASETS.IGNORE_LB
        ohem_thresh = cfg.LOSS.OHEM_THRESH

        with_datasets_aux = cfg.MODEL.WITH_DATASETS_AUX
        graph_node_features = gen_graph_node_feature(cfg)
        init_gnn_iters = cfg.MODEL.GNN.init_stage_iters
        Pretraining = cfg.MODEL.PRETRAINING
        gnn_iters = cfg.MODEL.GNN.GNN_ITERS
        seg_iters = cfg.MODEL.GNN.SEG_ITERS
        first_stage_gnn_iters = cfg.MODEL.GNN.FIRST_STAGE_GNN_ITERS
        num_unify_classes = cfg.DATASETS.NUM_UNIFY_CLASS
        with_spa_loss = cfg.LOSS.WITH_SPA_LOSS
        loss_weight_dict = cfg.LOSS.LOSS_WEIGHT_DICT
        
        return {
            'backbone': backbone,
            'sem_seg_head': sem_seg_head,
            'gnn_model': gnn_model,
            'datasets_cats': datasets_cats,
            'with_datasets_aux': with_datasets_aux, 
            'ignore_lb': ignore_lb,
            'ohem_thresh': ohem_thresh,
            "size_divisibility": cfg.MODEL.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "graph_node_features": graph_node_features,
            "init_gnn_iters": init_gnn_iters,
            "Pretraining": Pretraining,
            "gnn_iters": gnn_iters,
            "seg_iters": seg_iters,
            "first_stage_gnn_iters": first_stage_gnn_iters,
            "num_unify_classes": num_unify_classes,
            "with_spa_loss": with_spa_loss,
            "loss_weight_dict": loss_weight_dict
        }


    def forward(self, batched_inputs, dataset=0):
        # images = [x["image"].cuda() for x in batched_inputs]
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # images = ImageList.from_tensors(images, self.size_divisibility)
        # targets = [x["sem_seg"].cuda() for x in batched_inputs]
        # targets = self.prepare_targets(targets, images)
        # targets = torch.cat(targets, dim=0)
        # features = self.backbone(images.tensor)
        
        # if self.training:
        #     images = batched_inputs['image'].cuda()
        #     targets = batched_inputs['sem_seg'].cuda()
        #     features = self.backbone(images)  
        # else:

            
        iters = 0
        with EventStorage():
            storage = get_event_storage()
            iters = storage.iter


        
        images = [x["image"].cuda() for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        if self.training:
            images = ImageList.from_tensors(images, self.size_divisibility)
        else:
            images = ImageList.from_tensors(images, -1)
        targets = [x["sem_seg"].cuda() for x in batched_inputs]
        targets = self.prepare_targets(targets, images)
        targets = torch.cat(targets, dim=0)
        if self.training:
            dataset_lbs = [x["dataset_id"] for x in batched_inputs]
            dataset_lbs = torch.tensor(dataset_lbs).cuda()
        else:
            dataset_lbs = batched_inputs[0]["dataset_id"]
        
        if self.Pretraining:
            features = self.backbone(images.tensor)
            outputs = self.proj_head(features, dataset_lbs)

            if self.training:
                            # bipartite matching-based loss
                losses = {}
                self.alter_iters += 1
                for id, logit in enumerate(outputs['logits']):
                    logits = F.interpolate(logit, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    loss = self.criterion(logits, targets[dataset_lbs==id])
                    losses[f'loss_ce{id}'] = loss
                
                # for k in list(losses.keys()):
                #     if k in self.criterion.weight_dict:
                #         losses[k] *= self.criterion.weight_dict[k]
                #     else:
                #         # remove this loss if not specified in `weight_dict`
                #         losses.pop(k)
                return losses
            else:
                
                logits = F.interpolate(outputs['logits'], size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                processed_results = [{"sem_seg": logits[i]} for i in range(logits.shape[0])]
                return processed_results
        else:
            self.env_init(iters)
    
            if self.training:

                features = self.backbone(images.tensor)
                outputs = self.proj_head(features, dataset_lbs)
                unify_prototype, bi_graphs, _, _ = self.gnn_model(self.graph_node_features)
                losses = {}
                self.alter_iters += 1
                if self.train_seg_or_gnn == self.GNN:
                    if self.with_datasets_aux:
                        logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'], unify_prototype[:self.num_unify_classes])
                    else:
                        logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'], unify_prototype)
                else:
                    logits = outputs['logits']
                    if self.with_datasets_aux:
                        aux_logits_out = outputs['aux_logits']
                    
                # remap_logits = []
                uot_rate = np.min(int(self.alter_iters) / self.first_stage_gnn_iters, 1)
                adj_rate = 1 - uot_rate
                cur_cat = 0
                for i in range(self.n_datasets):
                    cur_cat += self.datasets_cats[i]
                    if not (dataset_lbs == i).any():
                        continue
                    if len(bi_graphs) == 2*self.n_datasets:
                        remap_logits_1 = torch.einsum('bchw, nc -> bnhw', logits[dataset_lbs==i], self.bipartite_graphs[2*i])
                        remap_logits_2 = torch.einsum('bchw, nc -> bnhw', logits[dataset_lbs==i], self.bipartite_graphs[2*i+1])
                    
                        remap_logits_1 = F.interpolate(remap_logits_1, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                        loss_1 = self.criterion(remap_logits_1, targets[dataset_lbs==i])
                        
                        remap_logits_2 = F.interpolate(remap_logits_2, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                        loss_2 = self.criterion(remap_logits_2, targets[dataset_lbs==i])
                        losses[f'loss_ce{i}'] = uot_rate*loss_1 + adj_rate*loss_2
                    else:
                        remap_logits = torch.einsum('bchw, nc -> bnhw', logits[dataset_lbs==i], self.bipartite_graphs[i])
                    
                        remap_logits = F.interpolate(remap_logits, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                        loss = self.criterion(remap_logits, targets[dataset_lbs==i])
                        
                        losses[f'loss_ce{i}'] = loss
            

                    if self.with_datasets_aux:
                        if self.train_seg_or_gnn == self.GNN:
                            aux_logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'][dataset_lbs==i], unify_prototype[self.num_unify_classes+cur_cat-self.datasets_cats[i]:self.num_unify_classes+cur_cat])
                        else:
                            aux_logits = aux_logits_out[dataset_lbs==i]
                        
                        aux_loss = self.criterion(aux_logits, targets[dataset_lbs==i])
                        losses[f'loss_aux{i}'] = aux_loss
                
                if self.with_spa_loss and self.train_seg_or_gnn == self.GNN and self.inFirstGNNStage:
                    if len(bi_graphs)==2*self.n_datasets:
                        spa_loss = torch.pow(torch.norm(bi_graphs[2*i+1], p='fro'), 2)
                    else:
                        spa_loss =  torch.pow(torch.norm(bi_graphs[i], p='fro'), 2)
                    
                    losses[f'loss_spa'] = spa_loss
                for k in list(losses.keys()):
                    if k in self.loss_weight_dict:
                        losses[k] *= self.loss_weight_dict[k]
                return losses
            else:
                self.backbone.eval()
                self.proj_head.eval()
                self.gnn_model.eval()
                
                features = self.backbone(images.tensor)
                outputs = self.proj_head(features, dataset_lbs)
                unify_prototype, bi_graphs, _, _ = self.gnn_model(self.graph_node_features)
                if self.train_seg_or_gnn == self.SEG:
                    logits = F.interpolate(outputs['logits'], size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    processed_results = [{"sem_seg": logits[i]} for i in range(logits.shape[0])]
                else:
                    if self.with_datasets_aux:
                        logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'], unify_prototype[:self.num_unify_classes])
                    else:
                        logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'], unify_prototype)
                    if len(bi_graphs) == 2*self.n_datasets:
                        logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[2*dataset_lbs])
                    else:
                        logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset_lbs])
                    logits = F.interpolate(logits, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    processed_results = [{"sem_seg": logits[i]} for i in range(logits.shape[0])]
                return processed_results                

    def env_init(self, iters):
        if self.initial == False:
            logger.info(f"initial: train_seg_or_gnn: {self.train_seg_or_gnn}")
            if self.train_seg_or_gnn == self.GNN:
                self.backbone.req_grad(False)
                self.proj_head.req_grad(False)
                self.gnn_model.req_grad(True)
                self.backbone.eval()
                self.proj_head.eval()
                self.gnn_model.train()
                if iters < self.init_gnn_iter:
                    self.gnn_model.frozenAdj(True)
            else:
                self.backbone.req_grad(True)
                self.proj_head.req_grad(True)
                self.gnn_model.req_grad(False)
                self.backbone.train()
                self.proj_head.train()
                self.gnn_model.eval()                    
            self.initial = True

        if self.train_seg_or_gnn == self.GNN:
            if self.inFirstGNNStage and int(self.alter_iters) > self.first_stage_gnn_iters:
                logger.info(f"change to second_gnn_stage")
                self.gnn_model.frozenAdj(True)
                self.inFirstGNNStage = False
            if int(self.alter_iters) > self.gnn_iters:
                logger.info(f"change to seg_stage")
                self.train_seg_or_gnn = self.SEG
                self.backbone.req_grad(True)
                self.proj_head.req_grad(True)
                self.gnn_model.req_grad(False)
                self.backbone.train()
                self.proj_head.train()
                self.gnn_model.eval() 
                self.alter_iters = torch.zeros(1)
        else:
            if int(self.alter_iters) > self.seg_iters:
                logger.info(f"change to gnn_stage")
                self.train_seg_or_gnn = self.GNN
                self.backbone.req_grad(False)
                self.proj_head.req_grad(False)
                self.gnn_model.req_grad(True)
                self.backbone.eval()
                self.proj_head.eval()
                self.gnn_model.train()
                self.alter_iters = torch.zeros(1)
                unify_prototype, bi_graphs, _, _ = self.gnn_model(self.graph_node_features)
                self.proj_head.set_bipartite_graphs(bi_graphs)
                self.proj_head.set_unify_prototype(unify_prototype, grad=True)
                    
                self.inFirstGNNStage = True
                
                    

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            # logger.info(f"image shape : {images.tensor.shape}, target shape : {targets_per_image.shape}")
            gt_masks = targets_per_image
            padded_masks = 255*torch.ones((1, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[0], : gt_masks.shape[1]] = gt_masks
            new_targets.append(
                padded_masks
            )
        return new_targets

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            # elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            #     if hasattr(module, 'last_bn') and module.last_bn:
            #         nn.init.zeros_(module.weight)
            #     else:
            #         nn.init.ones_(module.weight)
            #     nn.init.zeros_(module.bias)
        for name, param in self.named_parameters():
            if name.find('affine_weight') != -1:
                if hasattr(param, 'last_bn') and param.last_bn:
                    nn.init.zeros_(param)
                else:
                    nn.init.ones_(param)
            elif name.find('affine_bias') != -1:
                nn.init.zeros_(param)
                        
        # self.load_pretrain()

        
    def load_pretrain(self):
        state = torch.load(backbone_url)
        self.backbone.load_state_dict(state, strict=False)

    def get_params(self):
        def add_param_to_list(param, wd_params, nowd_params):
            # for param in mod.parameters():
            if param.requires_grad == False:
                return
                # continue
            
            if param.dim() == 1:
                nowd_params.append(param)
            elif param.dim() == 4 or param.dim() == 2:
                wd_params.append(param)
            else:
                nowd_params.append(param)
                print(param.dim())
                # print(param)
                print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        # for name, child in self.named_children():
        for name, param in self.named_parameters():
            
            if 'head' in name or 'aux' in name:
                add_param_to_list(param, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(param, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
    
    def set_bipartite_graphs(self, bi_graphs):
        
        if len(bi_graphs) == 2 * self.n_datasets:
            for i in range(0, self.n_datasets):
                self.bipartite_graphs[i] = nn.Parameter(
                    bi_graphs[2*i], requires_grad=False
                    )
        else:
            # print("bi_graphs len:", len(bi_graphs))
            for i in range(0, self.n_datasets):
                # print("i: ", i)
                self.bipartite_graphs[i] = nn.Parameter(
                    bi_graphs[i], requires_grad=False
                    )
            
        
    def set_unify_prototype(self, unify_prototype, grad=False):
        if self.with_datasets_aux and unify_prototype.shape[0] != self.unify_prototype.shape[0]:
            self.unify_prototype.data = unify_prototype[self.total_cats:]
            self.unify_prototype.requires_grad=grad
            cur_cat = 0
            for i in range(self.n_datasets):
                self.aux_prototype[i].data = unify_prototype[cur_cat:cur_cat+self.datasets_cats[i]]
                cur_cat += self.datasets_cats[i]
                self.aux_prototype[i].requires_grad=grad
        else:
            self.unify_prototype.data = unify_prototype
            self.unify_prototype.requires_grad=grad

        
    def get_encode_lb_vec(self):
        text_feature_vecs = []
        with torch.no_grad():
            clip_model, _ = clip.load("ViT-B/32", device="cuda")
            for i in range(0, self.n_datasets):
                lb_name = self.configer.get("dataset"+str(i+1), "label_names")
                lb_name = [f'a photo of {name} from dataset {i+1}.' for name in lb_name]
                text = clip.tokenize(lb_name).cuda()
                text_features = clip_model.encode_text(text).type(torch.float32)
                text_feature_vecs.append(text_features)
                
        text_feature_vecs = torch.cat(text_feature_vecs, dim=0)
        self.unify_prototype.data = text_feature_vecs
        self.unify_prototype.requires_grad=False
                