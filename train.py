import os
import time
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import build_dataset, build_dataloader
from model import build_model
from loss import build_loss, LossLog
from optim import build_optimizer, build_lr_scheduler
from metric import MetricLog, compute_metrics
from utils import visualization, write_scalar_to_tensorboard, write_pc_to_tensorboard, save_model, set_random_seed, load_cfg_file, make_dirs, summary_results, to_cuda

torch.autograd.set_detect_anomaly(True)

def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--config', required=True, help='the config file path')
    parser.add_argument('--pretrain', required=False, help='the pretrain ckpt path')
    
    args = parser.parse_args()
    return args

def train_step(train_loader, model, optimizer, loss_func, epoch, total_epoch, writer, vis_items):
    model.train()
    train_loss = LossLog()
    train_metrics = MetricLog()
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    loop.set_description(f'Epoch [{epoch}/{total_epoch}]')
    for idx, data_batch in loop:
        optimizer.zero_grad()
        to_cuda(data_batch)
        intput_data = model.module.create_input(data_batch)
        predictions = model(intput_data)
        ground_truth = model.module.create_ground_truth(data_batch)
        loss = loss_func(predictions, ground_truth)
        loss["loss"].backward()
        train_loss.add_loss(loss)
        optimizer.step()

        loop.set_postfix(loss = loss["loss"].item())
        """
        for i in range(pc_label.shape[0]):
            if (pc_label[i].item(), pc_id[i].item()) in vis_items:
                write_pc_to_tensorboard(writer, "train_vis", [source_pc[i], template_pc[i], pred_template_pcs[-1][i]], epoch)
        """
    results = summary_results("train", train_metrics, train_loss)
    write_scalar_to_tensorboard(writer, results, epoch)
    return results

def val_step(val_loader, model, loss_func, val_step, writer):
    model.eval()
    val_loss = LossLog()
    val_metrics = MetricLog()
    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader))
        loop.set_description(f'Val [{val_step}]')
        for idx, data_batch in loop:
            to_cuda(data_batch)
            intput_data = model.module.create_input(data_batch)
            predictions = model(intput_data)
            ground_truth = model.module.create_ground_truth(data_batch)

            minibatch_metrics = compute_metrics(predictions, ground_truth)
            val_metrics.add_metrics(minibatch_metrics)

    results = summary_results("val", val_metrics, val_loss)
    write_scalar_to_tensorboard(writer, results, val_step)
    return results

def main():
    args = config_params()
    cfg = load_cfg_file(args.config)
    cfg.dataloader.train_loader.batch_size *= torch.cuda.device_count()
    print(cfg)
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    experiment_dir = os.path.join(cfg.experiment_name, timestamp)
    make_dirs(experiment_dir)
    set_random_seed(cfg.seed)

    train_set = build_dataset(cfg.dataset.train_set)
    val_set = build_dataset(cfg.dataset.val_set)

    train_loader = build_dataloader(train_set, True, cfg.dataloader.train_loader)
    val_loader = build_dataloader(val_set, False, cfg.dataloader.val_loader)
    
    model = build_model(cfg.model)
    
    if args.pretrain:
        weights = torch.load(args.pretrain)
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace("module.", "") if "module" in k else  k
            weights_dict[new_k] = v

        model.load_state_dict(weights_dict, strict=False)
    
    if "overlap_pretrain_ckpt" in cfg:
        weights = torch.load(cfg.overlap_pretrain_ckpt)
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace("module.", "") if "module" in k else k
            weights_dict[new_k] = v

        model.overlap_predictor.load_state_dict(weights_dict)
    
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    loss_func = build_loss(cfg.loss)
    loss_func = loss_func.cuda()
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_lr_scheduler(optimizer, cfg.lr_scheduler) if "lr_scheduler" in cfg else None

    summary_path = os.path.join("exp", experiment_dir, "summary")
    writer = SummaryWriter(summary_path)
    
    checkpoints_path = os.path.join("exp", experiment_dir, "checkpoints")
    min_val_R_mse_error = float('inf')

    train_vis_items = [(label, id) for label, id in zip(cfg.train_vis_labels, cfg.train_vis_ids)]

    for epoch in range(cfg.epoch):
        train_results = train_step(train_loader, model, optimizer, loss_func, epoch + 1, cfg.epoch, writer, train_vis_items)     
        print(train_results)
        if scheduler:
            scheduler.step()
        
        if (epoch + 1)  % cfg.interval == 0:
            val_results = val_step(val_loader, model, loss_func, (epoch + 1) // cfg.interval, writer)
            print(val_results)
            save_model(checkpoints_path, "epoch_{}".format(str(epoch + 1)), model.state_dict())

            if "metrics_val/R_rmse" in val_results:
                val_R_error = val_results['metrics_val/R_rmse']
                if val_R_error < min_val_R_mse_error:
                    save_model(checkpoints_path, "min_val_R_rmse_error", model.state_dict())
                    min_val_R_mse_error = val_R_error
    
    writer.close()

if __name__ == "__main__":
    main()