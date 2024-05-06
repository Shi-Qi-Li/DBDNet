import os
import time
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import build_dataset, build_dataloader
from model import build_model
from metric import MetricLog, compute_metrics
from utils import write_scalar_to_tensorboard, set_random_seed, load_cfg_file, make_dirs, summary_results, overlap_visualization, to_cuda

def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--config', required=True, help='the config file path')
    parser.add_argument('--ckpt', required=True, help='the model ckpt path')
    
    args = parser.parse_args()
    return args

def test_step(test_loader, model, writer):
    model.eval()
    test_metrics = MetricLog()
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        loop.set_description(f'Test')
        for idx, data_batch in loop:
            to_cuda(data_batch)
            intput_data = model.module.create_input(data_batch)
            predictions = model(intput_data)
            ground_truth = model.module.create_ground_truth(data_batch)

            minibatch_metrics = compute_metrics(predictions, ground_truth)
            test_metrics.add_metrics(minibatch_metrics)
            
    results = summary_results("test", test_metrics, None)
    write_scalar_to_tensorboard(writer, results, 1)
    return results


def main():
    args = config_params()
    cfg = load_cfg_file(args.config)
    print(cfg)
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    experiment_dir = os.path.join(cfg.experiment_name, "test", timestamp)
    make_dirs(experiment_dir, mode="test")
    set_random_seed(cfg.seed)

    test_set = build_dataset(cfg.dataset.test_set)

    test_loader = build_dataloader(test_set, False, cfg.dataloader.test_loader)

    model = build_model(cfg.model)
    weights = torch.load(args.ckpt, map_location="cpu")
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace("module.", "") if "module" in k else  k
        weights_dict[new_k] = v

    model.load_state_dict(weights_dict)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    summary_path = os.path.join("exp", experiment_dir, "summary")
    writer = SummaryWriter(summary_path)

    test_results = test_step(test_loader, model, writer)
    print(test_results)
    
    writer.close()

if __name__ == "__main__":
    main()