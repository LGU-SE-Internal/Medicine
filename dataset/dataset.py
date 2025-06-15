from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Any, Optional
from joblib import Memory


class RCABenchDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        transform: Optional[Callable[[Path], tuple[Any, Any]]] = None,
        cache_dir: str = "./cache",
    ):
        self.data_packs = paths
        self.transform = transform

        memory = Memory(cache_dir, verbose=2)
        if self.transform:
            self._cached_transform = memory.cache(self.transform)
        else:
            self._cached_transform = None

    def __len__(self):
        return len(self.data_packs)

    def __getitem__(self, idx):
        data_pack = self.data_packs[idx]
        if self._cached_transform:
            return self._cached_transform(data_pack)
        return data_pack


def derive_filename(data_pack: Path):
    return {
        "abnormal_log": data_pack / "abnormal_logs.parquet",
        "normal_log": data_pack / "normal_logs.parquet",
        "abnormal_metric": data_pack / "abnormal_metrics.parquet",
        "normal_metric": data_pack / "normal_metrics.parquet",
        "abnormal_trace": data_pack / "abnormal_traces.parquet",
        "normal_trace": data_pack / "normal_traces.parquet",
        "env": data_pack / "env.json",
        "injection": data_pack / "injection.json",
    }
