from .dataset import RCABenchDataset, derive_filename
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from .utils import load_injection_data


class TraceDataset(RCABenchDataset):
    def __init__(self, paths: list[Path], cache_dir: str = "./cache"):
        super().__init__(paths, transform=self.transform_trace, cache_dir=cache_dir)
        self.transform = self.transform_trace

    def transform_trace(self, data_pack: Path) -> tuple[Any, Any]:
        """Transform a data pack to a tuple of (X, y).

        Args:
            data_pack (Path): Path to the data pack.

        Returns:
            tuple[Any, Any]: A tuple of (X, y) where X is the input data and y is the label.
        """
        fs = derive_filename(data_pack)

        # 1. 数据加载与预处理
        abnormal_trace_df = pd.read_parquet(fs["abnormal_trace"])
        normal_trace_df = pd.read_parquet(fs["normal_trace"])

        fault_type, target_service = load_injection_data(str(fs["injection"]))

        # 2. 时间戳处理（转换为秒）
        abnormal_trace_df["timestamp"] = abnormal_trace_df["time"].apply(
            lambda x: int(x.timestamp())
            if hasattr(x, "timestamp")
            else int(x / 1000000)
        )  # 处理Timestamp对象或纳秒值
        normal_trace_df["timestamp"] = normal_trace_df["time"].apply(
            lambda x: int(x.timestamp())
            if hasattr(x, "timestamp")
            else int(x / 1000000)
        )

        # 3. 构建调用链路
        # 通过parent_span_id和span_id建立调用关系
        abnormal_trace_df = self._build_invoke_links(abnormal_trace_df)
        normal_trace_df = self._build_invoke_links(normal_trace_df)

        # 4. 构建关键数据结构
        invoke_list = self._get_invoke_list(
            pd.concat([normal_trace_df, abnormal_trace_df])
        )

        # 构建实例列表（与 trace.py 逻辑一致）
        instance_list = list(
            set(abnormal_trace_df["service_name"].tolist()).union(
                set(normal_trace_df["service_name"].tolist())
            )
        )
        instance_list.sort()

        # 5. 计算参考基线（使用正常数据）
        ref_mean, ref_std = self._calculate_baseline(normal_trace_df, invoke_list)

        # 6. Z-Score异常检测和特征提取
        X = self._extract_features(abnormal_trace_df, invoke_list, ref_mean, ref_std)

        return X, {
            "fault_type": fault_type,
            "target_service": target_service,
        }

    def _build_invoke_links(self, df):
        """构建调用链路"""
        df = df.copy()

        # 创建服务调用映射
        span_to_service = dict(zip(df["span_id"], df["service_name"]))

        # 为每个span找到其父span的服务名
        df["parent_service"] = df["parent_span_id"].map(span_to_service)

        # 构建调用链路（父服务_子服务）
        df["invoke_link"] = (
            df["parent_service"].fillna("ROOT") + "_" + df["service_name"]
        )

        # 过滤掉根节点调用
        df = df[df["parent_service"].notna()]

        return df

    def _get_invoke_list(self, df):
        """获取所有调用链路列表"""
        return list(df["invoke_link"].unique())

    def _get_instance_list(self, df):
        """获取所有服务实例列表"""
        return list(df["service_name"].unique())

    def _calculate_baseline(self, normal_df, invoke_list):
        """计算参考基线"""
        ref_mean = {}
        ref_std = {}

        for invoke in invoke_list:
            invoke_data = normal_df[normal_df["invoke_link"] == invoke]
            if len(invoke_data) > 0:
                durations = invoke_data["duration"].values
                ref_mean[invoke] = np.mean(durations)
                ref_std[invoke] = np.std(durations)
            else:
                ref_mean[invoke] = 0
                ref_std[invoke] = 1

        return ref_mean, ref_std

    def _extract_features(self, abnormal_df, invoke_list, ref_mean, ref_std):
        """提取特征"""
        # 创建时间窗口（假设每分钟为一个窗口）
        abnormal_df["time_window"] = abnormal_df["timestamp"] // 60  # 每分钟一个窗口

        time_windows = sorted(abnormal_df["time_window"].unique())

        # 存储每个时间窗口的特征
        features = []

        for window in time_windows:
            window_data = abnormal_df[abnormal_df["time_window"] == window]
            window_features = np.zeros(len(invoke_list))

            # 计算每个调用链路的异常程度
            invoke_anomaly_scores = {}
            invoke_counts = {}

            for invoke in invoke_list:
                invoke_data = window_data[window_data["invoke_link"] == invoke]

                if len(invoke_data) > 0:
                    durations = invoke_data["duration"].values
                    mean = ref_mean[invoke]
                    std = ref_std[invoke]

                    # Z-Score异常检测
                    if std > 0:
                        z_scores = np.abs((durations - mean) / std)
                        anomalies = z_scores > 3  # 阈值为3

                        if np.any(anomalies):
                            # 异常Z-Score的平均值
                            invoke_anomaly_scores[invoke] = np.mean(z_scores[anomalies])
                            invoke_counts[invoke] = np.sum(anomalies)
                        else:
                            invoke_anomaly_scores[invoke] = 0
                            invoke_counts[invoke] = 0
                    else:
                        invoke_anomaly_scores[invoke] = 0
                        invoke_counts[invoke] = 0
                else:
                    invoke_anomaly_scores[invoke] = 0
                    invoke_counts[invoke] = 0

            # 计算动态权重
            cnt_list = np.array(
                [invoke_counts.get(invoke, 0) for invoke in invoke_list]
            )
            # 归一化处理，与 trace.py 保持一致
            cnt_list = (cnt_list - cnt_list.min()) / (
                cnt_list.max() - cnt_list.min() + 0.00001
            )
            cnt_list = np.log(cnt_list + 1)  # 对数变换
            cnt_diff = np.abs(
                np.concatenate([[0], np.diff(cnt_list)])
            )  # 差分计算变化率

            if len(cnt_diff) > 0 and cnt_diff.max() > np.mean(cnt_diff):
                total_gap = cnt_diff.max() - np.mean(cnt_diff)
            else:
                total_gap = 0.00001  # 修改为与 trace.py 一致的小值

            # 计算加权特征
            for i, invoke in enumerate(invoke_list):
                if total_gap > 0:
                    weight = cnt_diff[i] if i < len(cnt_diff) else 0
                    anomaly_score = invoke_anomaly_scores.get(invoke, 0)
                    window_features[i] = weight * anomaly_score / total_gap
                else:
                    window_features[i] = invoke_anomaly_scores.get(invoke, 0)

            features.append(window_features)

        return np.array(features) if features else np.array([[0] * len(invoke_list)])
