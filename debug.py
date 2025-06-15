from pathlib import Path
from dataset.dataset_log import LogDataset
from dataset.dataset_metric import MetricDataset
import os

paths = [
    Path(
        "/mnt/jfs/rcabench-platform-v2/data/rcabench_with_issues/ts3-ts-auth-service-response-replace-code-ns4vtw"
    )
]
dataset = MetricDataset(paths)
for i in range(len(dataset)):
    print(dataset[i])
    break
