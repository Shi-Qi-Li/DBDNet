from typing import List, Dict

import numpy as np

class LossLog: 
    loss_data: Dict[str, List]
    def __init__(self):
        self.loss_data = dict()

    def add_loss(self, loss: Dict):
        for key, val in loss.items():
            if key not in self.loss_data:
                self.__add_loss_category(key)

            self.loss_data[key].append(np.expand_dims(val.cpu().detach().numpy(), axis=0))

    def __add_loss_category(self, key: str):
        self.loss_data[key] = []

    def get_loss(self, key: str):
        return np.mean(np.concatenate(self.loss_data[key], axis=0))

    @property
    def all_loss_categories(self):
        return self.loss_data.keys()