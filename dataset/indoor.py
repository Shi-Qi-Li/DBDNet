import os
import numpy as np
import pickle

from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from utils import random_select_points
from .builder import DATASET

@DATASET
class IndoorReg(Dataset):
    def __init__(self, data_path: str, split: str = "train", sample_num: int = 2048, preload: bool = False):
        super(IndoorReg, self).__init__()
        assert split in ["train", "valid", "3DMatch", "3DLoMatch"], "Invalid data split!"

        self.preload = preload
        self.data_path = data_path
        self.sample_num = sample_num

        info_file = os.path.join(data_path, '{}.pkl'.format(split))
        with open(info_file, "rb") as f:
            info_data = pickle.load(f)

            self.R = info_data["rot"]
            self.t = info_data["trans"]
            self.source_path = info_data["src"]
            self.template_path = info_data["tgt"]

            self.len = len(info_data["rot"])
            self.overlap = info_data["overlap"]
        
        mask_file = os.path.join(data_path, '{}_overlap_info.pkl'.format(split))
        with open(mask_file, "rb") as f:
            info_data = pickle.load(f)

            self.source_mask = info_data["source_mask"]
            self.template_mask = info_data["template_mask"]
        
        if self.preload:    
            data_loop = tqdm(range(self.len))
            data_loop.set_description(f'Load data')
            
            self.source = []
            self.template = []
            for idx in data_loop:
                self.source.append(torch.load(f = os.path.join(data_path, self.source_path[idx])))
                self.template.append(torch.load(f = os.path.join(data_path, self.template_path[idx])))
                
            # self.source = [torch.load(f = os.path.join(data_path, pc_path)) for pc_path in self.source_path]
            # self.template = [torch.load(f = os.path.join(data_path, pc_path)) for pc_path in self.template_path]
            

    def __getitem__(self, index):
        
        if self.preload:
            source_pc = self.source[index].astype(np.float32)
            template_pc = self.template[index].astype(np.float32)
        else:
            source_pc = torch.load(f = os.path.join(self.data_path, self.source_path[index])).astype(np.float32)
            template_pc = torch.load(f = os.path.join(self.data_path, self.template_path[index])).astype(np.float32)

        R = self.R[index].astype(np.float32)
        t = self.t[index].reshape(3,).astype(np.float32)
        source_mask = self.source_mask[index].astype(np.float32)
        template_mask = self.template_mask[index].astype(np.float32)

        # source_pc = source_pc[np.random.permutation(source_pc.shape[0])[:self.sample_num]]  
        # template_pc = template_pc[np.random.permutation(template_pc.shape[0])[:self.sample_num]]
        source_raw = source_pc.copy()
        template_raw = template_pc.copy()
        source_pc, source_idx = random_select_points(source_pc, self.sample_num, True)
        template_pc, template_idx = random_select_points(template_pc, self.sample_num, True)
        
        source_mask = source_mask[source_idx]
        template_mask = template_mask[template_idx]
        overlap = self.overlap[index].astype(np.float32)

        data_batch = {
            "template_pc": template_pc,
            "source_pc": source_pc,
            "R": R,
            "t": t,
            "template_mask": template_mask,
            "source_mask": source_mask,
            "template_raw": template_raw,
            "source_raw": source_raw,
            "overlap": overlap,
            "index": index
        }

        return data_batch
    
    def __len__(self):
        return self.len