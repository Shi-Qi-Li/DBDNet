from utils import Registry

LOSS = Registry("loss")

def build_loss(cfg):
    return LOSS.build(cfg)