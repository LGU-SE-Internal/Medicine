from sklearn.model_selection import train_test_split
import torch
from typing import Optional
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model.fusion import AdaFusion, ExperimentModel
from dataset.dataset_log import LogDataset
from dataset.dataset_metric import MetricDataset
from dataset.dataset_trace import TraceDataset
from pathlib import Path


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MultiModalTrainer:
    def __init__(
        self,
        data_paths: list[Path],
        config: Optional[dict] = None,
        cache_dir: str = "./cache",
    ) -> None:
        self.data_paths = data_paths
        self.cache_dir = cache_dir
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.log_dataset = LogDataset(data_paths, cache_dir)
        self.metric_dataset = MetricDataset(data_paths, cache_dir)
        self.trace_dataset = TraceDataset(data_paths, cache_dir)

        self.log_data = []
        self.metric_data = []
        self.trace_data = []
        self.labels = []

        print("Loading data...")

        for i, data_pack in enumerate(tqdm(data_paths)):
            log_x, log_y = self.log_dataset[i]  # type:ignore
            metric_x, metric_y = self.metric_dataset[i]  # type:ignore
            trace_x, trace_y = self.trace_dataset[i]  # type:ignore

            self.log_data.append(log_x)
            self.metric_data.append(metric_x)
            self.trace_data.append(trace_x)

            self.labels.append(log_y["fault_type"])

        unique_labels = list(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = [self.label_to_idx[label] for label in self.labels]
        self.num_classes = len(unique_labels)

        print(f"Loaded {len(self.labels)} samples with {self.num_classes} classes")
        print(f"Unique labels: {unique_labels}")

        if self.log_data:
            sample_log = np.array(self.log_data[0])
            if sample_log.ndim == 1:
                self.log_feature_dim = len(sample_log)
            else:
                self.log_feature_dim = sample_log.shape[-1]
        else:
            self.log_feature_dim = 768

        if self.metric_data:
            sample_metric = np.array(self.metric_data[0])
            if sample_metric.ndim == 3:  # (services, time, features)
                self.metric_feature_dim = (
                    sample_metric.shape[0] * sample_metric.shape[2]
                )
            else:
                self.metric_feature_dim = int(np.prod(sample_metric.shape))
        else:
            self.metric_feature_dim = 100

        if self.trace_data:
            sample_trace = np.array(self.trace_data[0])
            self.trace_feature_dim = sample_trace.shape[-1]
        else:
            self.trace_feature_dim = 100

        print(
            f"Feature dimensions - Log: {self.log_feature_dim}, Metric: {self.metric_feature_dim}, Trace: {self.trace_feature_dim}"
        )

    def create_model(self):
        return AdaFusion(
            self.metric_feature_dim,
            self.trace_feature_dim,
            self.log_feature_dim,
            self.config.get("max_len", 512),
            self.config.get("d_model", 512),
            self.config.get("nhead", 8),
            self.config.get("d_ff", 2048),
            self.config.get("layer_num", 6),
            self.config.get("dropout", 0.1),
            self.num_classes,
            str(self.device),
        )

    def create_experiment_model(self):
        return ExperimentModel(
            self.metric_feature_dim,
            self.trace_feature_dim,
            self.log_feature_dim,
            self.config.get("max_len", 512),
            self.config.get("d_model", 512),
            self.config.get("nhead", 8),
            self.config.get("d_ff", 2048),
            self.config.get("layer_num", 6),
            self.config.get("dropout", 0.1),
            self.num_classes,
            str(self.device),
        )

    def get_data_loaders(self, batch_size: int = 32):
        # 划分数据集
        indices = list(range(len(self.labels)))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=self.labels
        )
        train_idx, eval_idx = train_test_split(
            train_idx, test_size=0.2, random_state=42
        )

        # 创建数据集
        train_data = []
        eval_data = []
        test_data = []

        for i in train_idx:
            train_data.append(
                (
                    self.log_data[i],
                    self.metric_data[i],
                    self.trace_data[i],
                    self.labels[i],
                )
            )

        for i in eval_idx:
            eval_data.append(
                (
                    self.log_data[i],
                    self.metric_data[i],
                    self.trace_data[i],
                    self.labels[i],
                )
            )

        for i in test_idx:
            test_data.append(
                (
                    self.log_data[i],
                    self.metric_data[i],
                    self.trace_data[i],
                    self.labels[i],
                )
            )

        print(
            f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}, Test samples: {len(test_data)}"
        )

        # 创建数据加载器
        train_dataset = CustomDataset(train_data)
        eval_dataset = CustomDataset(eval_data)
        test_dataset = CustomDataset(test_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        return train_loader, eval_loader, test_loader

    def collate_fn(self, batch):
        log_batch = []
        metric_batch = []
        trace_batch = []
        label_batch = []

        for log_data, metric_data, trace_data, label in batch:
            # 处理 log 数据
            log_array = np.array(log_data)
            if log_array.ndim == 2:  # (time_steps, features)
                # 使用平均池化或取最后一个时间步
                log_array = np.mean(log_array, axis=0)
            log_batch.append(log_array)

            # 处理 metric 数据
            metric_array = np.array(metric_data)
            if metric_array.ndim == 3:  # (services, time, features)
                # 展平为 (services * features)
                metric_array = metric_array.reshape(metric_array.shape[0], -1)
                metric_array = np.mean(metric_array, axis=0)  # 平均池化
            elif metric_array.ndim == 2:
                metric_array = np.mean(metric_array, axis=0)
            metric_batch.append(metric_array)

            # 处理 trace 数据
            trace_array = np.array(trace_data)
            if trace_array.ndim == 2:  # (time_steps, features)
                trace_array = np.mean(trace_array, axis=0)
            trace_batch.append(trace_array)

            label_batch.append(label)

        # 转换为张量
        log_tensor = torch.tensor(log_batch, dtype=torch.float32)
        metric_tensor = torch.tensor(metric_batch, dtype=torch.float32)
        trace_tensor = torch.tensor(trace_batch, dtype=torch.float32)
        label_tensor = torch.tensor(label_batch, dtype=torch.long)

        return (log_tensor, metric_tensor, trace_tensor), label_tensor

    def train(self, epochs: int = 100, batch_size: int = 32, lr: float = 1e-3):
        # 创建数据加载器
        train_loader, eval_loader, test_loader = self.get_data_loaders(batch_size)

        # 创建模型
        model = self.create_model()
        model.to(self.device)

        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        best_eval_loss = float("inf")
        best_model_state = None

        print("Starting training...")
        for epoch in tqdm(range(epochs), desc="Training"):
            model.train()
            total_loss = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # 将数据移到设备上
                log_input, metric_input, trace_input = inputs
                log_input = log_input.to(self.device)
                metric_input = metric_input.to(self.device)
                trace_input = trace_input.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                optimizer.zero_grad()
                outputs = model([log_input, metric_input, trace_input])
                loss = criterion(outputs, labels)

                # 反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # 验证
            eval_loss = self.evaluate(model, eval_loader, criterion)

            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}, Eval Loss: {eval_loss:.4f}"
            )

            # 保存最佳模型
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model_state = model.state_dict().copy()

        # 加载最佳模型并测试
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # 测试
        test_results = self.test(model, test_loader)
        return test_results

    def evaluate(self, model, eval_loader, criterion):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, labels in eval_loader:
                log_input, metric_input, trace_input = inputs
                log_input = log_input.to(self.device)
                metric_input = metric_input.to(self.device)
                trace_input = trace_input.to(self.device)
                labels = labels.to(self.device)

                outputs = model([log_input, metric_input, trace_input])
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(eval_loader)

    def test(self, model, test_loader):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                log_input, metric_input, trace_input = inputs
                log_input = log_input.to(self.device)
                metric_input = metric_input.to(self.device)
                trace_input = trace_input.to(self.device)

                outputs = model([log_input, metric_input, trace_input])
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # 计算指标
        metrics = self.calculate_metrics(all_labels, all_preds)
        return all_preds, all_labels, metrics

    def calculate_metrics(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        return {"precision": precision, "recall": recall, "f1": f1}
