from pathlib import Path
from dataset.dataset_log import LogDataset
from dataset.dataset_metric import MetricDataset
from dataset.dataset_trace import TraceDataset
import os

paths = [
    Path(
        "/mnt/jfs/rcabench-platform-v2/data/rcabench_with_issues/ts3-ts-auth-service-response-replace-code-ns4vtw"
    )
]
# dataset1 = TraceDataset(paths)
dataset2 = LogDataset(paths)
dataset3 = MetricDataset(paths)

# _ = dataset1[0]
# print("trace")

_ = dataset2[0]
print("log")
_ = dataset3[0]
print("metric")
