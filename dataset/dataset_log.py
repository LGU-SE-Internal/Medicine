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
import pickle
import os
from tqdm import tqdm
from .utils import load_injection_data


# 全局BERT编码器实例
_global_bert_encoder = None


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

        # 简单的磁盘缓存
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "sentence_embeddings.pkl"

        # 加载现有缓存
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        """从磁盘加载缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    cache = pickle.load(f)
                print(f"Loaded {len(cache)} cached embeddings from {self.cache_file}")
                return cache
            except Exception as e:
                print(f"Failed to load cache: {e}, starting with empty cache")
                return {}
        return {}

    def _save_cache(self):
        """保存缓存到磁盘"""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
            print(f"Saved {len(self._cache)} embeddings to cache")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def _get_cache_key(self, sentence: str, no_wordpiece: bool = False) -> str:
        """生成缓存键"""
        return f"{sentence}|{no_wordpiece}"

    @classmethod
    def get_global_instance(cls, cache_dir: str = "./cache/bert_encoder"):
        """获取全局BERT编码器实例，确保跨数据集共享缓存"""
        global _global_bert_encoder
        if _global_bert_encoder is None:
            _global_bert_encoder = cls(cache_dir)
        return _global_bert_encoder

    def _encode_single(self, sentence, no_wordpiece=False):
        """内部单个句子编码方法"""
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

    def _batch_encode_internal(self, sentences, no_wordpiece=False, batch_size=32):
        """内部批量编码方法"""
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
                embeddings.extend(batch_embeddings.cpu().numpy().tolist())

        return embeddings

    def encode(self, sentence, no_wordpiece=False):
        """单个句子编码，使用磁盘缓存"""
        cache_key = self._get_cache_key(sentence, no_wordpiece)

        # 检查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 计算嵌入
        embedding = self._encode_single(sentence, no_wordpiece)

        # 保存到缓存
        self._cache[cache_key] = embedding

        return embedding

    def batch_encode(
        self, sentences, no_wordpiece=False, batch_size=32, save_every=100
    ):
        """批量编码接口，使用磁盘缓存"""
        print(f"Encoding {len(sentences)} templates using disk cache...")

        # 分离已缓存和未缓存的句子
        cached_results = {}
        uncached_sentences = []
        uncached_indices = []

        for i, sentence in enumerate(sentences):
            cache_key = self._get_cache_key(sentence, no_wordpiece)
            if cache_key in self._cache:
                cached_results[i] = self._cache[cache_key]
            else:
                uncached_sentences.append(sentence)
                uncached_indices.append(i)

        print(
            f"Found {len(cached_results)} cached embeddings, computing {len(uncached_sentences)} new ones"
        )

        # 批量编码未缓存的句子
        if uncached_sentences:
            new_embeddings = self._batch_encode_internal(
                uncached_sentences, no_wordpiece, batch_size
            )

            # 保存新的嵌入到缓存
            for i, (sentence, embedding) in enumerate(
                zip(uncached_sentences, new_embeddings)
            ):
                cache_key = self._get_cache_key(sentence, no_wordpiece)
                self._cache[cache_key] = embedding
                cached_results[uncached_indices[i]] = embedding

                # 定期保存缓存
                if (i + 1) % save_every == 0:
                    self._save_cache()

        # 最终保存缓存
        self._save_cache()

        # 按原始顺序返回结果
        result = [cached_results[i] for i in range(len(sentences))]
        print("Batch encoding completed!")
        return result

    def __call__(self, sentence, no_wordpiece=False):
        """单个句子编码，使用磁盘缓存"""
        return self.encode(sentence, no_wordpiece)

    def get_cache_info(self):
        """获取缓存信息"""
        return {
            "cache_dir": str(self.cache_dir),
            "cache_file": str(self.cache_file),
            "cached_embeddings": len(self._cache),
        }

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        if self.cache_file.exists():
            os.remove(self.cache_file)
        print("Disk cache cleared!")

    def save_cache(self):
        """手动保存缓存"""
        self._save_cache()


class DrainProcesser:
    def __init__(
        self, conf: str, save_path: str, cache_dir: str = "./cache/drain"
    ) -> None:
        self._drain_config_path = conf
        persistence = FilePersistence(save_path)
        miner_config = TemplateMinerConfig()
        miner_config.load(self._drain_config_path)
        self._template_miner = TemplateMiner(persistence, config=miner_config)

        # 简单的磁盘缓存
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "sentence_templates.pkl"

        # 加载现有缓存
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        """从磁盘加载缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    cache = pickle.load(f)
                print(f"Loaded {len(cache)} cached templates from {self.cache_file}")
                return cache
            except Exception as e:
                print(f"Failed to load drain cache: {e}, starting with empty cache")
                return {}
        return {}

    def _save_cache(self):
        """保存缓存到磁盘"""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
            print(f"Saved {len(self._cache)} templates to cache")
        except Exception as e:
            print(f"Failed to save drain cache: {e}")

    def __call__(self, sentence) -> str:
        line = str(sentence).strip()

        # 检查缓存
        if line in self._cache:
            return self._cache[line]

        # 使用drain处理
        result = self._template_miner.add_log_message(line)
        if "template_mined" not in result:
            raise KeyError(f"'template_mined' key not found in result: {result}")

        template = result["template_mined"]

        # 保存到缓存
        self._cache[line] = template

        return template

    def get_cache_info(self):
        """获取缓存信息"""
        return {
            "cache_dir": str(self.cache_dir),
            "cache_file": str(self.cache_file),
            "cached_templates": len(self._cache),
        }

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        if self.cache_file.exists():
            os.remove(self.cache_file)
        print("Drain cache cleared!")

    def save_cache(self):
        """手动保存缓存"""
        self._save_cache()


class LogDataset(RCABenchDataset):
    def __init__(self, paths: list[Path], cache_dir: str = "./cache"):
        super().__init__(paths, transform=self.transform_log, cache_dir=cache_dir)
        self._drain = DrainProcesser(
            "dataset/drain3/drain.ini", "data/gaia/drain.bin", f"{cache_dir}/drain"
        )
        self._encoder = BertEncoder.get_global_instance(f"{cache_dir}/bert_encoder")

    def transform_log(self, data_pack: Path) -> tuple[Any, Any]:
        """Transform a data pack to a tuple of (X, y).

        Args:
            data_pack (Path): Path to the data pack.

        Returns:
            tuple[Any, Any]: A tuple of (X, y) where X is the input data and y is the label.
        """
        fs = derive_filename(data_pack)

        # 检查必要的文件键是否存在
        if "abnormal_log" not in fs:
            raise KeyError(f"'abnormal_log' key not found in fs: {list(fs.keys())}")
        if "injection" not in fs:
            raise KeyError(f"'injection' key not found in fs: {list(fs.keys())}")

        df1 = pd.read_parquet(fs["abnormal_log"])

        fault_type, target_service = load_injection_data(str(fs["injection"]))

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
        print(f"Processing {len(all_templates_list)} unique templates...")

        # 使用批量编码方法，完全基于磁盘缓存
        template_embeddings_list = self._encoder.batch_encode(
            all_templates_list, batch_size=64
        )

        # 保存drain缓存
        self._drain.save_cache()

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
