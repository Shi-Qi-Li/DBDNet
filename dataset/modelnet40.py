import os
import numpy as np
import h5py
import json
import random

from torch.utils.data import Dataset

from utils import generate_random_rotation_matrix, generate_random_tranlation_vector, random_select_points, transform, jitter, random_crop, generate_overlap_mask, inv_R_t
from .builder import DATASET

@DATASET
class ModelNet40Reg(Dataset):
    def __init__(self, data_path, categories, train, mode="clean", normal=False, descriptor=None, sample_num=1024):
        super(ModelNet40Reg, self).__init__()
        self.categories = categories
        self.mode = mode
        self.sample_num = sample_num
        self.normal = normal
        
        if self.mode not in ["clean", "noise", "crop"]:
            raise NotImplementedError

        if train:
            files = [os.path.join(data_path, 'ply_data_train{}.h5'.format(i)) for i in range(5)]
            id_files = [os.path.join(data_path, 'ply_data_train_{}_id2file.json'.format(i)) for i in range(5)]
        else:
            files = [os.path.join(data_path, 'ply_data_test{}.h5'.format(i)) for i in range(2)]
            id_files = [os.path.join(data_path, 'ply_data_test_{}_id2file.json'.format(i)) for i in range(2)]

        self.data, self.label, self.id = self.load_from_h5(files, id_files)


    def load_from_h5(self, files, id_files):
        points, normals, labels, ids = [], [], [], []

        for file, id_file in zip(files, id_files):
            f = h5py.File(file, 'r')
            cur_points = f['data'][:].astype(np.float32)
            cur_normal = f['normal'][:].astype(np.float32)
            cur_label = f['label'][:].astype(np.int32)
            with open(id_file) as f:
                cur_id = json.load(f)
                cur_id = [int(_.split('/')[-1].split('.')[0].split('_')[-1]) for _ in cur_id]
                cur_id = np.asarray(cur_id, dtype=np.int32)

            mask = np.isin(cur_label, self.categories).flatten()
            cur_points = cur_points[mask, ...]
            cur_normal = cur_normal[mask, ...]
            cur_label = cur_label[mask, ...]
            cur_id = cur_id[mask, ...]

            points.append(cur_points)
            normals.append(cur_normal)
            labels.append(cur_label)
            ids.append(np.expand_dims(cur_id, axis=-1))
            
        points = np.concatenate(points, axis=0)
        normals = np.concatenate(normals, axis=0)
        label = np.concatenate(labels, axis=0)
        id = np.concatenate(ids, axis=0)
        if self.normal:
            data = np.concatenate([points, normals], axis=-1).astype(np.float32)
        else:
            data = points
        return data, label, id


    def __getitem__(self, index):        
        original_pc = self.data[index, ...]
        label = self.label[index, ...]
        id = self.id[index, ...]
   
        R = generate_random_rotation_matrix()
        t = generate_random_tranlation_vector()

        template_pc = random_select_points(original_pc, self.sample_num)
        
        if self.mode == "clean":
            source_pc = template_pc.copy()
        else:
            source_pc = random_select_points(original_pc, self.sample_num)
        
        source_points = transform(source_pc[:, :3], R, t)
            
        if self.normal:
            source_normals = transform(source_pc[:, 3:], R)
            source_pc = np.concatenate([source_points, source_normals], axis=-1)
        else:
            source_pc = source_points

        if self.mode == "crop":
            source_pc, source_crop_direction = random_crop(source_pc)
            template_pc, template_crop_direction = random_crop(template_pc)

        # TO DO: data jitter set
        if self.mode != "clean":
            source_pc[:, :3] = jitter(source_pc[:, :3])
            template_pc[:, :3] = jitter(template_pc[:, :3])        

        template_mask, source_mask = generate_overlap_mask(template_pc, source_pc, R, t)

        R, t = inv_R_t(R, t)

        data_batch = {
            "template_pc": template_pc,
            "source_pc": source_pc,
            "R": R,
            "t": t,
            "label": label,
            "id": id,
            "template_mask": template_mask,
            "source_mask": source_mask
        }

        return data_batch
    
    def __len__(self):
        return len(self.data)