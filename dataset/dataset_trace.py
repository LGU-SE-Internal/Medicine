from .dataset import RCABenchDataset, derive_filename
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from .utils import load_injection_data


class TraceDataset(RCABenchDataset):
    def __init__(self, paths: list[Path], cache_dir: str = "./cache"):
        super().__init__(
            paths,
            transform=self.transform_trace,
            cache_dir=cache_dir,
            cache_name="dataset_trace",
            use_dataset_cache=False,
        )

    def transform_trace(self, data_pack: Path) -> tuple[Any, Any]:
        fs = derive_filename(data_pack)

        abnormal_trace_df = pd.read_parquet(fs["abnormal_trace"])
        normal_trace_df = pd.read_parquet(fs["normal_trace"])

        fault_type, target_service = load_injection_data(str(fs["injection"]))

        abnormal_trace_df["timestamp"] = abnormal_trace_df["time"].apply(
            lambda x: int(x.timestamp())
            if hasattr(x, "timestamp")
            else int(x / 1000000)
        )
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
        cnt_of_invoke = {}
        for invoke in invoke_list:
            cnt_of_invoke[invoke] = []

        abnormal_df["time_window"] = abnormal_df["timestamp"] // 60
        time_windows = sorted(abnormal_df["time_window"].unique())

        all_window_features = []

        for window in time_windows:
            window_data = abnormal_df[abnormal_df["time_window"] == window]
            window_features = np.zeros(len(invoke_list))

            for i, invoke in enumerate(invoke_list):
                invoke_data = window_data[window_data["invoke_link"] == invoke]

                if len(invoke_data) > 0:
                    durations = invoke_data["duration"].values
                    mean = ref_mean[invoke]
                    std = ref_std[invoke]

                    if std > 0:
                        z_scores = np.abs((durations - mean) / std)
                        anomalies = int(np.sum(z_scores > 3))
                        cnt_of_invoke[invoke].append(anomalies)

                        if anomalies > 0:
                            window_features[i] = np.mean(z_scores[z_scores > 3])
                        else:
                            window_features[i] = 0
                    else:
                        window_features[i] = 0
                        cnt_of_invoke[invoke].append(0)
                else:
                    window_features[i] = 0
                    cnt_of_invoke[invoke].append(0)

            all_window_features.append(window_features)

        total_gap = 0.00001
        wei_of_invoke = {}
        for invoke, cnt_list in cnt_of_invoke.items():
            cnt_list = np.array(cnt_list)
            cnt_list = (cnt_list - cnt_list.min()) / (
                cnt_list.max() - cnt_list.min() + 0.00001
            )
            cnt_list = np.log(cnt_list + 1)
            cnt_list = np.abs([0] + list(np.diff(cnt_list)))
            gap = cnt_list.max() - np.mean(cnt_list)
            wei_of_invoke[invoke] = gap
            total_gap += gap

        # 聚合所有时间窗口的特征为单个一维向量
        if all_window_features:
            # 计算所有时间窗口的平均特征（或可以使用最大值、求和等聚合方式）
            aggregated_features = np.mean(all_window_features, axis=0)
        else:
            aggregated_features = np.zeros(len(invoke_list))

        # 应用权重
        weighted_features = np.zeros(len(invoke_list))
        for i, invoke in enumerate(invoke_list):
            weighted_features[i] = (
                wei_of_invoke[invoke] * aggregated_features[i] / total_gap
            )

        return weighted_features

    def get_feature_dim(self):
        return len(getattr(self, "invoke_list", []))

    def get_invoke_list(self):
        return getattr(self, "invoke_list", [])
