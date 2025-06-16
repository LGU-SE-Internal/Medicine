from .dataset import RCABenchDataset, derive_filename
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from .utils import load_injection_data, CacheManager
import hashlib
import json


class TraceDataset(RCABenchDataset):
    def __init__(self, paths: list[Path], cache_dir: str = "./cache"):
        self._global_invoke_cache = CacheManager[list](
            Path(cache_dir) / "global_invoke_list.pkl"
        )

        self.global_invoke_list = self._build_global_invoke_list(paths)
        print(
            f"Built global invoke_list with {len(self.global_invoke_list)} invoke links"
        )

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

        abnormal_trace_df = self._build_invoke_links(abnormal_trace_df)
        normal_trace_df = self._build_invoke_links(normal_trace_df)

        invoke_list = self.global_invoke_list

        instance_list = list(
            set(abnormal_trace_df["service_name"].tolist()).union(
                set(normal_trace_df["service_name"].tolist())
            )
        )
        instance_list.sort()

        ref_mean, ref_std = self._calculate_baseline(normal_trace_df, invoke_list)

        X = self._extract_features(abnormal_trace_df, invoke_list, ref_mean, ref_std)

        return X, {
            "fault_type": fault_type,
            "target_service": target_service,
        }

    def _build_invoke_links(self, df):
        df = df.copy()

        span_to_service = dict(zip(df["span_id"], df["service_name"]))

        df["parent_service"] = df["parent_span_id"].map(span_to_service)

        df["invoke_link"] = (
            df["parent_service"].fillna("ROOT") + "_" + df["service_name"]
        )

        df = df[df["parent_service"].notna()]

        return df

    def _get_invoke_list(self, df):
        return list(df["invoke_link"].unique())

    def _get_instance_list(self, df):
        return list(df["service_name"].unique())

    def _calculate_baseline(self, normal_df, invoke_list):
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
        for invoke in invoke_list:
            if invoke in cnt_of_invoke:
                cnt_list = np.array(cnt_of_invoke[invoke])
                if len(cnt_list) > 0:
                    cnt_list = (cnt_list - cnt_list.min()) / (
                        cnt_list.max() - cnt_list.min() + 0.00001
                    )
                    cnt_list = np.log(cnt_list + 1)
                    cnt_list = np.abs([0] + list(np.diff(cnt_list)))
                    gap = cnt_list.max() - np.mean(cnt_list)
                    wei_of_invoke[invoke] = gap
                    total_gap += gap
                else:
                    wei_of_invoke[invoke] = 0.0
            else:
                wei_of_invoke[invoke] = 0.0

        if all_window_features:
            aggregated_features = np.mean(all_window_features, axis=0)
        else:
            aggregated_features = np.zeros(len(invoke_list))

        weighted_features = np.zeros(len(invoke_list))
        for i, invoke in enumerate(invoke_list):
            weight = wei_of_invoke.get(invoke, 0.0)
            if total_gap > 0:
                weighted_features[i] = weight * aggregated_features[i] / total_gap
            else:
                weighted_features[i] = aggregated_features[i]

        return weighted_features

    def get_feature_dim(self):
        return len(getattr(self, "global_invoke_list", []))

    def get_invoke_list(self):
        return getattr(self, "global_invoke_list", [])

    def _build_global_invoke_list(self, paths: list[Path]) -> list[str]:
        all_invoke_links = set()
        for data_pack in paths:
            try:
                stat = data_pack.stat()
                key_data = {
                    "path": str(data_pack.resolve()),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            except FileNotFoundError:
                key_data = {
                    "path": str(data_pack.resolve()),
                    "size": -1,
                    "mtime": -1,
                }
            cache_key = hashlib.md5(
                json.dumps(key_data, sort_keys=True).encode()
            ).hexdigest()
            cached_invoke_list = self._global_invoke_cache.get(cache_key)
            if cached_invoke_list is not None:
                all_invoke_links.update(cached_invoke_list)
                continue
            try:
                fs = derive_filename(data_pack)
                abnormal_trace_df = pd.read_parquet(fs["abnormal_trace"])
                normal_trace_df = pd.read_parquet(fs["normal_trace"])
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
                abnormal_trace_df = self._build_invoke_links(abnormal_trace_df)
                normal_trace_df = self._build_invoke_links(normal_trace_df)
                pack_invoke_links = self._get_invoke_list(
                    pd.concat([normal_trace_df, abnormal_trace_df])
                )
                all_invoke_links.update(pack_invoke_links)
                self._global_invoke_cache.set(cache_key, pack_invoke_links)
                print(
                    f"Data pack {data_pack.name}: {len(pack_invoke_links)} invoke links (cached)"
                )
            except Exception as e:
                print(f"Warning: Failed to process {data_pack} for invoke_list: {e}")
                continue
        self._global_invoke_cache.save()
        global_invoke_list = sorted(list(all_invoke_links))
        print(
            f"Global invoke_list contains {len(global_invoke_list)} unique invoke links"
        )
        return global_invoke_list
