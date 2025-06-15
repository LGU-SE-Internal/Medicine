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
from tqdm import tqdm


class BertEncoder:
    def __init__(self, cache_dir: str = "./cache/bert_encoder") -> None:
        self._bert_tokenizer = BertTokenizer.from_pretrained(
            "google-bert/bert-base-uncased",
            cache_dir="./cache/bert",
            # local_files_only=True,
        )
        self._bert_model = BertModel.from_pretrained(
            "google-bert/bert-base-uncased",
            cache_dir="./cache/bert",
            # local_files_only=True,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._bert_model = self._bert_model.to(self.device)  # type: ignore
        self._bert_model.eval()  # 设置为评估模式

        self.cache = {}
        self.memory = Memory(cache_dir, verbose=0)
        self._cached_encode = self.memory.cache(self._encode)
        self._cached_batch_encode = self.memory.cache(self._batch_encode)

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
        # 将输入移动到设备上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():  # 禁用梯度计算以节省内存
            outputs = self._bert_model(**inputs)
            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze(dim=1)
        return embedding.cpu().numpy().tolist()

    def _batch_encode(self, sentences, no_wordpiece=False, batch_size=32):
        """批量编码句子列表"""
        if no_wordpiece:
            processed_sentences = []
            for sentence in sentences:
                words = sentence.split(" ")
                words = [
                    word for word in words if word in self._bert_tokenizer.vocab.keys()
                ]
                processed_sentences.append(" ".join(words))
            sentences = processed_sentences

        embeddings = []

        # 批量处理
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i : i + batch_size]

            # 批量tokenize
            inputs = self._bert_tokenizer(
                batch_sentences,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )

            # 将输入移动到设备上
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._bert_model(**inputs)
                # 对每个序列计算平均池化
                batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                embeddings.extend(batch_embeddings.cpu().numpy())

        return embeddings

    def batch_encode(self, sentences, no_wordpiece=False, batch_size=32):
        """公共批量编码接口，支持缓存"""
        # 将句子列表转换为tuple以便缓存
        cache_key = tuple(sentences)
        if cache_key in self.cache:
            return self.cache[cache_key]

        result = self._cached_batch_encode(sentences, no_wordpiece, batch_size)
        self.cache[cache_key] = result
        return result

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

        seqs = []
        cnt_of_log = {}

        for cnt, df in tqdm(
            enumerate(dfs_per_minute),
            desc="Processing drain templates",
            total=len(dfs_per_minute),
        ):
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
        # 收集所有唯一的模板，进行批量编码
        all_templates = set()
        for seq in seqs:
            all_templates.update(seq)

        # 批量编码所有模板
        all_templates_list = list(all_templates)
        print(f"Batch encoding {len(all_templates_list)} unique templates...")

        # 直接调用内部方法避免缓存问题
        template_embeddings_list = self._encoder._batch_encode(
            all_templates_list, batch_size=64
        )

        # 创建模板到嵌入的映射
        template_to_embedding = {}
        for i, template in enumerate(all_templates_list):
            template_to_embedding[template] = np.array(template_embeddings_list[i])

        # 现在快速生成序列嵌入
        new_seq = []
        for seq in tqdm(seqs, desc="Generating sequence embeddings"):
            repr = np.zeros((768,))
            for template in seq:
                repr += (
                    wei_of_log[template] * template_to_embedding[template] / total_gap
                )
            new_seq.append(repr.tolist())
        return new_seq, {
            "fault_type": fault_type,
            "target_service": target_service,
        }
