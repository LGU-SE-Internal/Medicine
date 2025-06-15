from .dataset import RCABenchDataset, derive_filename
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import json


class MetricDataset(RCABenchDataset):
    """
    Metric Dataset for RCABench format data processing.

    This dataset processes time-series metric data from microservice systems,
    extracting features for root cause analysis and anomaly detection.

    Features:
    - Time-based aggregation of metrics
    - Service-level feature extraction
    - Normalization and preprocessing
    - Support for both normal and abnormal metric data
    """

    def __init__(self, paths: list[Path], cache_dir: str = "./cache"):
        """
        Initialize MetricDataset

        Args:
            paths: List of data pack paths in RCABench format
            cache_dir: Directory for caching processed data
        """
        super().__init__(paths, transform=self.transform_metric, cache_dir=cache_dir)

    def transform_metric(self, data_pack: Path) -> tuple[Any, Any]:
        """Transform a data pack to a tuple of (X, y).

        Args:
            data_pack (Path): Path to the data pack.

        Returns:
            tuple[Any, Any]: A tuple of (X, y) where X is the input data and y is the label.
        """
        fs = derive_filename(data_pack)

        # 读取异常期间的指标数据
        df_abnormal = pd.read_parquet(fs["abnormal_metric"])

        # 读取注入信息获取标签
        with open(fs["injection"], "r") as f:
            injection = json.load(f)
            fault_type = injection["fault_type"]

            engine = json.loads(injection["display_config"])
            target_service = engine["injection_point"]["app_name"]

        df_abnormal = df_abnormal.sort_values(by="time")

        df_abnormal["time_bucket"] = pd.to_datetime(
            df_abnormal["time"], unit="s"
        ).dt.floor("min")

        all_services = df_abnormal["service_name"].unique().tolist()
        all_metrics = df_abnormal["metric_name"].unique().tolist()

        aggregated_data = []
        time_buckets = sorted(df_abnormal["time_bucket"].unique())

        for time_bucket in time_buckets:
            time_data = df_abnormal[df_abnormal["time_bucket"] == time_bucket]

            service_features = []
            for service in all_services:
                service_data = time_data[time_data["service_name"] == service]

                metric_features = []
                for metric in all_metrics:
                    metric_data = service_data[service_data["metric_name"] == metric]
                    if not metric_data.empty:
                        avg_value = metric_data["value"].mean()
                        normalized_value = self._normalize_value(avg_value)
                        metric_features.append(normalized_value)
                    else:
                        metric_features.append(0.0)

                service_features.append(metric_features)

            aggregated_data.append(service_features)

        X = np.array(aggregated_data)

        y = {
            "fault_type": fault_type,
            "target_service": target_service,
        }

        return X, y

    def _normalize_value(self, value):
        try:
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return np.tanh(value / 100.0)  # 假设指标值通常在 [0, 100] 范围内
        except (TypeError, ValueError):
            return 0.0
