from typing import Dict, Optional

import os
import torch
import yaml
import numpy as np
from easydict import EasyDict as edict
from open3d.visualization.tensorboard_plugin.util import to_dict_batch

from loss import LossLog
from metric import MetricLog
from utils import covert_to_pcd

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def write_scalar_to_tensorboard(writer, results, epoch):
    for key, value in results.items():
        if value != None:
            writer.add_scalar(key, value, epoch)

def write_pc_to_tensorboard(writer, title, pcs, epoch):
    source_pc = torch.squeeze(pcs[0]).cpu().numpy()
    template_pc = torch.squeeze(pcs[1]).cpu().numpy()
    pred_template_pc = torch.squeeze(pcs[2]).cpu().detach().numpy()

    source_pc = covert_to_pcd(source_pc, 0)
    template_pc = covert_to_pcd(template_pc, 1)
    pred_template_pc = covert_to_pcd(pred_template_pc, 2)

    writer.add_3d(title, to_dict_batch([source_pc, template_pc, pred_template_pc]), epoch)

def save_model(checkpoints_path, name, model_state):
    saved_path = os.path.join(checkpoints_path, "{}{}".format(name, ".pth")) 
    torch.save(model_state, saved_path)

def load_cfg_file(model_cfg_path):
    with open(model_cfg_path) as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    return edict(cfg)

def make_dirs(experiment_stamp, mode="train"):
    work_dir = os.path.join("exp", experiment_stamp)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    summary_dir = os.path.join(work_dir, "summary")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    if mode == "train":
        checkpoints_dir = os.path.join(work_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        
def summary_results(mode: str, metrics: MetricLog, loss: Optional[LossLog] = None):
    results = dict()

    for metric_category in metrics.all_metric_categories:
        results.update({''.join(["metrics_", mode, "/", metric_category]): metrics.get_metric(metric_category)})

    if mode != "test":
        for loss_category in loss.all_loss_categories:
            results.update({''.join(["loss_", mode, "/", loss_category]): loss.get_loss(loss_category)})

    return results

def to_cuda(data_batch: Dict):
    for key, value in data_batch.items():
        if isinstance(value, torch.Tensor):
            data_batch[key] = value.cuda()