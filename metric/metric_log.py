from typing import List, Dict

import sklearn
import numpy as np

class MetricLog: 
    metric_data: Dict[str, List]
    def __init__(self):
        self.metric_data = dict()

    def add_metrics(self, metrics: Dict):
        for key, val in metrics.items():
            if key not in self.metric_data:
                self.__add_metric_category(key)

            self.metric_data[key].append(val)

    def __add_metric_category(self, key: str):
        self.metric_data[key] = []

    def get_metric(self, key: str):
        if key == "R_rmse":
            metric = np.mean(np.concatenate(self.metric_data["R_mse"], axis=0))
            metric = np.sqrt(metric)
        elif key == "t_rmse":
            metric = np.mean(np.concatenate(self.metric_data["t_mse"], axis=0))
            metric = np.sqrt(metric)
        elif key == "accuracy":
            metric = (np.sum(np.concatenate(self.metric_data["tp"], axis=0)) + 
                np.sum(np.concatenate(self.metric_data["tn"], axis=0))) / (
                np.sum(np.concatenate(self.metric_data["tp"], axis=0)) + 
                np.sum(np.concatenate(self.metric_data["fp"], axis=0)) + 
                np.sum(np.concatenate(self.metric_data["fn"], axis=0)) + 
                np.sum(np.concatenate(self.metric_data["tn"], axis=0))
            )
        elif key == "precision":
            metric = np.sum(np.concatenate(self.metric_data["tp"], axis=0)) / (
                np.sum(np.concatenate(self.metric_data["tp"], axis=0)) + 
                np.sum(np.concatenate(self.metric_data["fp"], axis=0))
            )
        elif key == "recall":
            metric = np.sum(np.concatenate(self.metric_data["tp"], axis=0)) / (
                np.sum(np.concatenate(self.metric_data["tp"], axis=0)) + 
                np.sum(np.concatenate(self.metric_data["fn"], axis=0))
            )
        elif key == "f1_score":
            metric = 2 * np.sum(np.concatenate(self.metric_data["tp"], axis=0)) / (
                2 * np.sum(np.concatenate(self.metric_data["tp"], axis=0)) + 
                np.sum(np.concatenate(self.metric_data["fp"], axis=0)) + 
                np.sum(np.concatenate(self.metric_data["fn"], axis=0))
            )
        elif key == "OA":
            metric = 100.0 * sklearn.metrics.accuracy_score(
                np.concatenate(self.metric_data["label"], axis=0),
                np.concatenate(self.metric_data["pred"], axis=0))
        elif key == "mAcc":
            metric = 100.0 * sklearn.metrics.balanced_accuracy_score(
                np.concatenate(self.metric_data["label"], axis=0),
                np.concatenate(self.metric_data["pred"], axis=0))
        elif key == "registration_recall":
            metric = np.mean(
                np.logical_and(
                    np.concatenate(self.metric_data["R_isotropic"], axis=0) < 15,
                    np.concatenate(self.metric_data["t_isotropic"], axis=0) < 0.3
                )
            )
        else:
            metric = np.mean(np.concatenate(self.metric_data[key], axis=0))

        return metric
 
    @property
    def all_metric_categories(self):
        metric_categories = list(self.metric_data.keys())
        if "R_mse" in metric_categories:
            metric_categories[metric_categories.index("R_mse")] = "R_rmse"
        if "t_mse" in metric_categories:
            metric_categories[metric_categories.index("t_mse")] = "t_rmse"
        
        if all([category in metric_categories for category in ["tp", "fp", "fn", "tn"]]):
            metric_categories.remove("tp")
            metric_categories.remove("fp")
            metric_categories.remove("fn")
            metric_categories.remove("tn")

            metric_categories.append("accuracy")
            metric_categories.append("precision")
            metric_categories.append("recall")
            metric_categories.append("f1_score")
        
        if all([category in metric_categories for category in ["pred", "label"]]):
            metric_categories.remove("pred")
            metric_categories.remove("label")

            metric_categories.append("OA")
            metric_categories.append("mAcc")

        if all([category in metric_categories for category in ["R_isotropic", "t_isotropic"]]):
            metric_categories.append("registration_recall")

        return metric_categories