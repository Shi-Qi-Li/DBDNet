from .modelnet40 import ModelNet40Reg
from .indoor import IndoorReg
from .builder import DATASET, build_dataset, build_dataloader

__all__ = ["DATASET", "build_dataset", "build_dataloader"]