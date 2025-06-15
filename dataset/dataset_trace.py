from .dataset import RCABenchDataset, derive_filename
from pathlib import Path
from typing import Any
from transformers import BertTokenizer, BertModel
import torch
from .drain3.file_persistence import FilePersistence
from .drain3.template_miner import TemplateMiner
from .drain3.template_miner_config import TemplateMinerConfig
import pandas as pd
import numpy as np
import json
from joblib import Memory


class BertEncoder:
    def __init__(self, cache_dir: str = "./cache/bert_encoder") -> None:
        self._bert_tokenizer = BertTokenizer.from_pretrained(
            "google-bert/bert-base-uncased",
            cache_dir="./cache/bert",
            local_files_only=True,
        )
        self._bert_model = BertModel.from_pretrained(
            "google-bert/bert-base-uncased",
            cache_dir="./cache/bert",
            local_files_only=True,
        )
        self.cache = {}
        self.memory = Memory(cache_dir, verbose=0)
        self._cached_encode = self.memory.cache(self._encode)

    def _encode(self, sentence, no_wordpiece=False):
        if no_wordpiece:
            words = sentence.split(" ")
            words = [
                word for word in words if word in self._bert_tokenizer.vocab.keys()
            ]
            sentence = " ".join(words)
        inputs = self._bert_tokenizer(
            sentence, truncation=True, return_tensors="pt", max_length=512
        )
        outputs = self._bert_model(**inputs)
        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze(dim=1)
        return embedding[0].tolist()

    def __call__(self, sentence, no_wordpiece=False):
        result = self._cached_encode(sentence, no_wordpiece)
        self.cache[sentence] = result
        return result


class DrainProcesser:
    def __init__(self, conf: str, save_path: str) -> None:
        self._drain_config_path = conf
        persistence = FilePersistence(save_path)
        miner_config = TemplateMinerConfig()
        miner_config.load(self._drain_config_path)
        self._template_miner = TemplateMiner(persistence, config=miner_config)

    def __call__(self, sentence) -> str:
        line = str(sentence).strip()
        result = self._template_miner.add_log_message(line)
        return result["template_mined"]


class LogDataset(RCABenchDataset):
    def __init__(self, paths: list[Path], cache_dir: str = "./cache"):
        super().__init__(paths, transform=self.transform_log, cache_dir=cache_dir)
        self._drain = DrainProcesser("dataset/drain3/drain.ini", "data/gaia/drain.bin")
        self._encoder = BertEncoder()
        self.transform = self.transform_log

    def transform_log(self, data_pack: Path) -> tuple[Any, Any]:
        """Transform a data pack to a tuple of (X, y).

        Args:
            data_pack (Path): Path to the data pack.

        Returns:
            tuple[Any, Any]: A tuple of (X, y) where X is the input data and y is the label.
        """
        fs = derive_filename(data_pack)
        df1 = pd.read_parquet(fs["abnormal_log"])

        with open(fs["injection"], "r") as f:
            injection = json.load(f)
            fault_type = injection["fault_type"]

            engine = json.loads(injection["display_config"])
            target_service = engine["injection_point"]["app_name"]

        # 1. 排序
        df1 = df1.sort_values(by="time")
        # 2. 转换为 datetime 并 floor 到分钟级别（用于分组）
        df1["time_bucket"] = pd.to_datetime(df1["time"], unit="s").dt.floor("min")
        # 3. 分组（每分钟为一个 group），存成一个 list
        grouped = df1.groupby("time_bucket")
        # 4. 每分钟一个 DataFrame，存到列表中
        dfs_per_minute = [group.copy() for _, group in grouped]
        print(dfs_per_minute[0])

        seqs = []
        cnt_of_log = {}

        for cnt, df in enumerate(dfs_per_minute):
            log_templates = []
            for log in df["message"].tolist():
                template = self._drain(log)
                log_templates.append(template)
                if cnt_of_log.get(template, None) is None:
                    cnt_of_log[template] = [0] * len(dfs_per_minute)
                cnt_of_log[template][cnt] += 1
            seqs.append(list(set(log_templates)))
        wei_of_log = {}
        total_gap = 0.00001
        for template, cnt_list in cnt_of_log.items():
            cnt_list = np.array(cnt_list)
            cnt_list = np.log(cnt_list + 0.00001)
            cnt_list = np.abs([0] + np.diff(cnt_list))
            gap = cnt_list.max() - cnt_list.mean()
            wei_of_log[template] = gap
            total_gap += gap
        new_seq = []
        for seq in seqs:
            repr = np.zeros((768,))
            for template in seq:
                repr += (
                    wei_of_log[template] * np.array(self._encoder(template)) / total_gap
                )
            new_seq.append(repr.tolist())
        return new_seq, {
            "fault_type": fault_type,
            "target_service": target_service,
        }
