import torch
import inspect

from utils import Registry

OPTIMIZER = Registry("optimizer")
LR_SCHEDULER = Registry("lr_scheduler")

def register_torch_optimizers():
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        optim = getattr(torch.optim, module_name)
        if inspect.isclass(optim) and issubclass(optim, torch.optim.Optimizer):
            OPTIMIZER.register(optim)

def register_torch_lr_scheduler():
    for module_name in dir(torch.optim.lr_scheduler):
        if module_name.startswith('__') or module_name.startswith('_'):
            continue
        lr_scheduler = getattr(torch.optim.lr_scheduler, module_name)
        if inspect.isclass(lr_scheduler) and issubclass(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            LR_SCHEDULER.register(lr_scheduler)
        
register_torch_optimizers()
register_torch_lr_scheduler()

def build_optimizer(model, cfg):
    cfg["params"] = filter(lambda param: param.requires_grad, model.parameters())
    return OPTIMIZER.build(cfg)

def build_lr_scheduler(optimizer, cfg):
    cfg["optimizer"] = optimizer
    return LR_SCHEDULER.build(cfg)