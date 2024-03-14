
import sys

sys.path.insert(0, '.')
import os
import os.path as osp


import logging
import argparse
import math

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import ExitStack, contextmanager


from detectron2.engine.hooks import HookBase
import datetime
from detectron2.utils.logger import log_every_n_seconds
from .MCMF_build_for_unseen import MinCostMaxFlow_Unseen

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

def build_bipartite_graph_for_unseen(trainer_build_test_loader, cfg, model):
    """
    Build a bipartite graph for unseen classes.

    Args:
        cfg (CfgNode): config.
        model (nn.Module): model.

    Returns:
        tuple: (graph, num_classes).
    """
    logger = logging.getLogger(__name__)

    # org_aux = net.aux_mode

    bipart_graph = model.get_bipart_graph()
    
    datasets_cats = cfg.DATASETS.DATASETS_CATS
    n_datasets = len(datasets_cats)
    ignore_index = cfg.DATASETS.IGNORE_LB
    total_cats = 0
    callbacks = None
    for i in range(0, n_datasets):
        total_cats += datasets_cats[i]
    num_unfiy_class = cfg.DATASETS.NUM_UNIFY_CLASS
    target_bipart = []

    for dataset_idx, dataset_name in enumerate(cfg.DATASETS.TRAIN):
        data_loader = trainer_build_test_loader(cfg, dataset_name)

        n_classes = datasets_cats[dataset_idx]
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
                    labels = [x["sem_seg"][None].cuda() for x in inputs]

                    logits = [output["uni_logits"] for output in outputs]
                    
                    for lb, lg in zip(labels, logits):
                        lg = lg[None]
                        lb = lb.long()
                        # lb = F.interpolate(lb.unsqueeze(1).float(), size=(lg.shape[2], lg.shape[3]),
                        #         mode='nearest').squeeze(1).long()

                        probs = torch.softmax(lg, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        logger = logging.getLogger(__name__) 
                        keep = lb != ignore_index
                        # logger.info(f"lb:{lb.shape}, keep:{keep.shape}, preds:{preds.shape}")

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
                "Total inference time: {} ({:.6f} s / iter per device)".format(
                    total_time_str, total_time / (total - num_warmup)
                )
            )
            total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
            logger.info(
                "Total inference pure compute time: {} ({:.6f} s / iter per device)".format(
                    total_compute_time_str, total_compute_time / (total - num_warmup)
                )
            )                    
            
        cost_matrix = torch.zeros(num_unfiy_class, n_classes)
        for uni_class in range(num_unfiy_class):
            total_pres = torch.sum(hist[:, uni_class])
            for set_class in range(n_classes):
                total_target = torch.sum(hist[set_class])
                iou = hist[set_class, uni_class] / (total_pres + total_target - hist[set_class, uni_class])
                cost_matrix[uni_class, set_class] = 1 - iou
                
        mcmf = MinCostMaxFlow_Unseen()
        src, target = mcmf(cost_matrix) 
        
        # torch.set_printoptions(profile="full")
        # print(hist)

        bipart = torch.zeros(n_classes, num_unfiy_class)
        for s, t in zip(src, target):
            bipart[t,s] = 1
            # buckets[index] = new_val
        target_bipart.append(bipart)

    # net.train()
    # target_bipart.cat(target_bipart, dim=0)
    model.set_target_bipart(target_bipart)
    # return graph, num_classes