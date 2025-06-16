from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Any, Optional
from joblib import Memory
import hashlib
import json
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .utils import CacheManager


class RCABenchDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        transform: Optional[Callable[[Path], tuple[Any, Any]]] = None,
        cache_dir: str = "./cache",
        cache_name: str = "dataset",
        use_dataset_cache: bool = False,
        max_workers: Optional[int] = None,
    ):
        self.data_packs = paths
        self.transform = transform
        self.cache_dir = cache_dir
        self.use_dataset_cache = use_dataset_cache

        if use_dataset_cache:
            # Use dataset-level caching (similar to LogDataset)
            self._dataset_cache = CacheManager[tuple](
                Path(cache_dir) / "dataset" / f"{cache_name}.pkl"
            )
            self._processed_data: list[tuple] = []

            if max_workers is None:
                max_workers = min(multiprocessing.cpu_count(), len(paths), 32)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {
                    executor.submit(self._process_path, path): path for path in paths
                }

                with tqdm(
                    total=len(paths), desc=f"Preprocessing {cache_name} datasets"
                ) as pbar:
                    for future in future_to_path:
                        try:
                            result = future.result()
                            if result:
                                self._processed_data.append(result)
                        except Exception as e:
                            path_name = future_to_path[future].name
                            logging.error(
                                f"Error processing {path_name}: {e}", exc_info=True
                            )
                        finally:
                            pbar.update(1)

            logging.info(
                f"Preprocessing complete. {len(self._processed_data)}/{len(paths)} data packs processed successfully."
            )
            self._save_all_caches()
        else:
            # Use original joblib caching
            self._processed_data = []
            memory = Memory(cache_dir, verbose=0)
            if self.transform:
                self._cached_transform = memory.cache(self.transform)
            else:
                self._cached_transform = None

    def _save_all_caches(self):
        """A single place to save all underlying caches."""
        if self.use_dataset_cache:
            self._dataset_cache.save()

    @staticmethod
    def _get_cache_key(data_pack: Path) -> str:
        """Generates a cache key based on file metadata."""
        try:
            stat = data_pack.stat()
            key_data = {
                "path": str(data_pack.resolve()),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
        except FileNotFoundError:
            logging.warning(
                f"Could not stat file for cache key: {data_pack}. Using path hash."
            )
            return hashlib.md5(str(data_pack.resolve()).encode()).hexdigest()

    def _process_path(self, data_pack: Path) -> Optional[tuple]:
        """Processes a single data pack, using the cache."""
        key = self._get_cache_key(data_pack)

        def compute_fn():
            result = self.transform(data_pack) if self.transform else None
            if result is None:
                raise ValueError(f"Transform returned None for {data_pack}")
            return result

        processed_item = self._dataset_cache.get_or_compute(key, compute_fn)
        return processed_item

    def __len__(self):
        if self.use_dataset_cache:
            return len(self._processed_data)
        return len(self.data_packs)

    def __getitem__(self, idx):
        if self.use_dataset_cache:
            if not 0 <= idx < len(self._processed_data):
                raise IndexError(
                    f"Index {idx} is out of range for the dataset of size {len(self)}."
                )
            return self._processed_data[idx]
        else:
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
