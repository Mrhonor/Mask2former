# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config, add_hrnet_config, add_gnn_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
    MaskFormerSemanticDatasetMapper_2,
)
from .data.dataset_mappers.semantic_dataset_mapper import SemanticDatasetMapper

# models
from .maskformer_model import MaskFormer
from .HRNetv2_model import HRNet_W48_ARCH
from .HRNetv2_model_finetune import HRNet_W48_Finetune_ARCH
from .HRNetv2_model_finetune_unseen import HRNet_W48_Finetune_Unseen_ARCH
from .HRNetv2_model_finetune_vis import HRNet_W48_Finetune_Vis_ARCH
from .HRNetv2_model_mseg_eval import HRNet_W48_Mseg_ARCH
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .data.dataloader.DaliDataLoader import DaLiLoaderAdapter, LoaderAdapter
from .utils.build_bipartite_graph_for_unseen import build_bipartite_graph_for_unseen, build_bipartite_graph_for_unseen_for_manually
from .utils.eval_mseg import eval_for_mseg_datasets
from .utils.save_result import save_result
from .utils.find_specific_class import find_specific_class