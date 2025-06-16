from .dataset import RCABenchDataset, derive_filename
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from .utils import load_injection_data


class MetricDataset(RCABenchDataset):
    def __init__(self, paths: list[Path], cache_dir: str = "./cache"):
        super().__init__(
            paths,
            transform=self.transform_metric,
            cache_dir=cache_dir,
            cache_name="dataset_metric",
            use_dataset_cache=True,
        )

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
        fault_type, target_service = load_injection_data(str(fs["injection"]))

        df_abnormal = df_abnormal.sort_values(by="time")

        df_abnormal["time_bucket"] = pd.to_datetime(
            df_abnormal["time"], unit="s"
        ).dt.floor("min")

        df_abnormal["service_name"] = (
            df_abnormal["service_name"]
            .fillna(df_abnormal["attr.k8s.deployment.name"])
            .combine_first(df_abnormal["attr.k8s.statefulset.name"])
        )
        df_abnormal = df_abnormal.dropna(subset=["service_name"])

        all_services = df_abnormal["service_name"].unique().tolist()
        all_metrics = df_abnormal["metric"].unique().tolist()

        # Update metric and service lists for feature tracking
        self.metric_list = sorted(list(set(all_metrics)))
        self.service_list = sorted(list(set(all_services)))
        self.metric_num = len(self.metric_list)

        metric_norm_params = self._calculate_metric_normalization_params(
            df_abnormal, all_metrics
        )

        time_buckets = sorted(df_abnormal["time_bucket"].unique())

        aggregated_data = []

        for service in all_services:
            service_time_series = []

            for time_bucket in time_buckets:
                time_data = df_abnormal[df_abnormal["time_bucket"] == time_bucket]
                service_data = time_data[time_data["service_name"] == service]

                metric_features = []
                for metric in all_metrics:
                    metric_data = service_data[service_data["metric"] == metric]
                    if not metric_data.empty:
                        avg_value = metric_data["value"].mean()
                        normalized_value = self._normalize_value_with_params(
                            avg_value, metric_norm_params[metric]
                        )
                        metric_features.append(normalized_value)
                    else:
                        metric_features.append(0.0)

                service_time_series.append(metric_features)

            # 应用时间序列差分处理（与原始metric.py逻辑一致）
            service_time_series = self._apply_time_series_diff(service_time_series)

            aggregated_data.append(service_time_series)

        X = np.array(aggregated_data)

        y = {
            "fault_type": fault_type,
            "target_service": target_service,
        }

        return X, y

    def _calculate_metric_normalization_params(self, df, all_metrics):
        metric_params = {}

        for metric in all_metrics:
            metric_data = df[df["metric"] == metric]["value"]

            if not metric_data.empty:
                valid_data = metric_data.dropna()
                valid_data = valid_data[np.isfinite(valid_data)]

                if not valid_data.empty:
                    metric_params[metric] = {
                        "min": valid_data.min(),
                        "max": valid_data.max(),
                        "mean": valid_data.mean(),
                        "std": valid_data.std(),
                    }
                else:
                    metric_params[metric] = {
                        "min": 0.0,
                        "max": 1.0,
                        "mean": 0.0,
                        "std": 1.0,
                    }
            else:
                metric_params[metric] = {
                    "min": 0.0,
                    "max": 1.0,
                    "mean": 0.0,
                    "std": 1.0,
                }

        return metric_params

    def _normalize_value_with_params(self, value, params):
        try:
            if np.isnan(value) or np.isinf(value):
                return 0.0

            # 使用与原始metric.py一致的z-score标准化方法
            if params["std"] > 0:
                z_score = (value - params["mean"]) / (
                    params["std"] + 0.00001
                )  # 避免除0
                return z_score
            else:
                if params["max"] > params["min"]:
                    return (value - params["min"]) / (params["max"] - params["min"])
                else:
                    return 0.0
        except (TypeError, ValueError):
            return 0.0

    def _apply_time_series_diff(self, service_time_series):
        """Apply first-order difference to time series data, consistent with metric.py logic"""
        if len(service_time_series) <= 1:
            return service_time_series

        # 对每个指标维度分别计算差分
        diff_series = []
        for i, time_step in enumerate(service_time_series):
            if i == 0:
                # 第一个时间步设为0（与原始逻辑一致）
                diff_series.append([0.0] * len(time_step))
            else:
                # 计算与前一时间步的差分
                prev_step = service_time_series[i - 1]
                curr_diff = []
                for j in range(len(time_step)):
                    diff_val = time_step[j] - prev_step[j]
                    curr_diff.append(diff_val)
                diff_series.append(curr_diff)

        return diff_series
