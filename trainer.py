from sklearn.model_selection import train_test_split
import torch
from typing import Optional, Literal, Tuple
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model.fusion import AdaFusion, ExperimentModel
from dataset.dataset_log import LogDataset
from dataset.dataset_metric import MetricDataset
from dataset.dataset_trace import TraceDataset
from pathlib import Path
import math
import time
import random


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
        # 配置logger
        logger.add("debug.log", rotation="500 MB", level="DEBUG")
        logger.info("Initializing MultiModalTrainer")

        self.data_paths = data_paths
        self.cache_dir = cache_dir
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")
        logger.info(f"Number of data paths: {len(data_paths)}")

        self.trace_dataset = TraceDataset(data_paths, cache_dir)
        self.log_dataset = LogDataset(data_paths, cache_dir)
        self.metric_dataset = MetricDataset(data_paths, cache_dir)

        self.log_data = []
        self.metric_data = []
        self.trace_data = []
        self.labels = []

        print("Loading data...")

        for i, data_pack in enumerate(tqdm(data_paths)):
            log_x, log_y = self.log_dataset[i]  # type:ignore
            metric_x, metric_y = self.metric_dataset[i]  # type:ignore
            trace_x, trace_y = self.trace_dataset[i]  # type:ignore

            # 添加详细维度调试信息
            log_shape = np.array(log_x).shape if log_x is not None else "None"
            metric_shape = np.array(metric_x).shape if metric_x is not None else "None"
            trace_shape = np.array(trace_x).shape if trace_x is not None else "None"

            logger.debug(f"Sample {i} from {data_pack}:")
            logger.debug(f"  log_x: shape={log_shape}, type={type(log_x)}")
            logger.debug(f"  metric_x: shape={metric_shape}, type={type(metric_x)}")
            logger.debug(f"  trace_x: shape={trace_shape}, type={type(trace_x)}")
            logger.debug(f"  fault_type: {log_y['fault_type']}")

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

        # 执行全面的维度检查
        log_consistent, metric_consistent, trace_consistent = (
            self.check_data_dimensions("after data loading")
        )

        if not trace_consistent:
            logger.error(
                "Training will likely fail due to inconsistent trace dimensions!"
            )
            logger.error("Consider preprocessing the data to ensure consistent shapes.")

        if self.log_data:
            sample_log = np.array(self.log_data[0])
            if sample_log.ndim == 1:
                self.log_feature_dim = len(sample_log)
            else:
                self.log_feature_dim = sample_log.shape[-1]

        if self.metric_data:
            sample_metric = np.array(self.metric_data[0])
            if sample_metric.ndim == 3:  # (services, time, features)
                self.metric_feature_dim = sample_metric.shape[2]  # 只取特征维度
            else:
                self.metric_feature_dim = sample_metric.shape[-1]

            self.instance_num = sample_metric.shape[0]

        if self.trace_data:
            sample_trace = np.array(self.trace_data[0])
            self.trace_feature_dim = sample_trace.shape[-1]

        print(
            f"Feature dimensions - Log: {self.log_feature_dim}, Metric: {self.metric_feature_dim}, Trace: {self.trace_feature_dim}"
        )

    def create_model(self):
        return AdaFusion(
            self.metric_feature_dim,  # kpi_num
            self.trace_feature_dim,  # invoke_num
            self.instance_num,
            512,
            768,
            8,
            256,
            2,
            0.3,
            self.num_classes,
            str(self.device),
        )

    def create_experiment_model(self):
        return ExperimentModel(
            self.metric_feature_dim,
            self.trace_feature_dim,
            self.instance_num,
            512,
            768,
            8,
            256,
            2,
            0.3,
            self.num_classes,
            str(self.device),
        )

    def get_data_loaders(self, batch_size: int = 32):
        logger.info(f"Creating data loaders with batch_size={batch_size}")

        # 使用原始的数据分割方式
        X_train, X_test, y_train, y_test = train_test_split(
            range(len(self.labels)), self.labels, test_size=0.2, random_state=42
        )
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        data = list(zip(self.log_data, self.metric_data, self.trace_data, self.labels))

        train_data = []
        for index in X_train:
            train_data.append(data[index])

        print(
            f"Train samples: {len(train_data)}, Eval samples: {len(X_eval)}, Test samples: {len(X_test)}"
        )

        # 检查训练数据的维度一致性
        logger.info("Checking train data dimensions...")
        train_trace_shapes = [np.array(item[2]).shape for item in train_data]
        if len(set(train_trace_shapes)) > 1:
            logger.error(
                f"Train data has inconsistent trace shapes: {set(train_trace_shapes)}"
            )

        train_dataset = CustomDataset(train_data)

        # 为eval和test创建正确的数据子集
        eval_data = [data[i] for i in X_eval]
        test_data = [data[i] for i in X_test]

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

        logger.debug(f"Collating batch of size: {len(batch)}")

        for i, (log_data, metric_data, trace_data, label) in enumerate(batch):
            log_batch.append(log_data)
            metric_batch.append(metric_data)
            trace_batch.append(trace_data)
            label_batch.append(label)

            # 记录每个样本的详细维度信息
            log_shape = np.array(log_data).shape if log_data is not None else "None"
            metric_shape = (
                np.array(metric_data).shape if metric_data is not None else "None"
            )
            trace_shape = (
                np.array(trace_data).shape if trace_data is not None else "None"
            )

            logger.debug(f"  Batch item {i}:")
            logger.debug(f"    log: {log_shape} (type: {type(log_data)})")
            logger.debug(f"    metric: {metric_shape} (type: {type(metric_data)})")
            logger.debug(f"    trace: {trace_shape} (type: {type(trace_data)})")
            logger.debug(f"    label: {label}")

        # 检查所有样本的维度一致性
        log_shapes = [np.array(log).shape for log in log_batch if log is not None]
        metric_shapes = [
            np.array(metric).shape for metric in metric_batch if metric is not None
        ]
        trace_shapes = [
            np.array(trace).shape for trace in trace_batch if trace is not None
        ]

        logger.debug("Batch shapes summary:")
        logger.debug(f"  Log shapes: {log_shapes}")
        logger.debug(f"  Metric shapes: {metric_shapes}")
        logger.debug(f"  Trace shapes: {trace_shapes}")

        # 检查维度不一致的情况
        if len(set(log_shapes)) > 1:
            logger.error("BATCH ERROR: Log data dimension mismatch in batch!")
            for i, shape in enumerate(log_shapes):
                logger.error(f"  Log item {i}: {shape}")

        if len(set(metric_shapes)) > 1:
            logger.error("BATCH ERROR: Metric data dimension mismatch in batch!")
            for i, shape in enumerate(metric_shapes):
                logger.error(f"  Metric item {i}: {shape}")

        if len(set(trace_shapes)) > 1:
            logger.error("BATCH ERROR: Trace data dimension mismatch in batch!")
            for i, shape in enumerate(trace_shapes):
                logger.error(f"  Trace item {i}: {shape}")

        try:
            logger.debug("Creating tensors...")
            log_tensor = torch.tensor(log_batch, dtype=torch.float32)
            metric_tensor = torch.tensor(metric_batch, dtype=torch.float32)
            trace_tensor = torch.tensor(trace_batch, dtype=torch.float32)

            logger.debug("Tensor creation successful!")
            logger.debug(
                f"  Final tensor shapes: log={log_tensor.shape}, "
                f"metric={metric_tensor.shape}, trace={trace_tensor.shape}"
            )

            return (
                (log_tensor, metric_tensor, trace_tensor),
                torch.tensor(label_batch, dtype=torch.long),
            )
        except Exception as e:
            logger.error(f"TENSOR CREATION FAILED: {e}")
            logger.error("Attempting to create tensors from:")
            logger.error(f"  log_batch shapes: {log_shapes}")
            logger.error(f"  metric_batch shapes: {metric_shapes}")
            logger.error(f"  trace_batch shapes: {trace_shapes}")

            # 尝试找出问题所在
            for i, (log, metric, trace) in enumerate(
                zip(log_batch, metric_batch, trace_batch)
            ):
                try:
                    torch.tensor([log], dtype=torch.float32)
                    torch.tensor([metric], dtype=torch.float32)
                    torch.tensor([trace], dtype=torch.float32)
                except Exception as item_error:
                    logger.error(f"  Item {i} failed: {item_error}")
                    logger.error(f"    log shape: {np.array(log).shape}")
                    logger.error(f"    metric shape: {np.array(metric).shape}")
                    logger.error(f"    trace shape: {np.array(trace).shape}")

            raise

    def sigmoid(self, data):
        """Sigmoid函数"""
        fz = []
        for num in data:
            fz.append(1 / (1 + math.exp(-num)))
        return fz

    def train(self, epochs: int = 100, batch_size: int = 32, lr: float = 1e-3):
        train_loader, eval_loader, test_loader = self.get_data_loaders(batch_size)

        model = self.create_model()
        model.to(self.device)

        # 使用原始的优化器配置
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        criterion = torch.nn.CrossEntropyLoss()

        best_model = (torch.inf, None)

        print("Starting training...")
        for epoch in tqdm(range(epochs), desc="Training"):
            model.train()

            if epoch < 1 or epoch > 80:
                # 简单训练阶段
                avg_loss = 0
                train_m_loss = 0
                train_l_loss = 0
                train_t_loss = 0

                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    try:
                        inputs = [_inputs.to(self.device) for _inputs in inputs]
                        labels = labels.to(self.device)

                        logger.debug(
                            f"Epoch {epoch}, Batch {batch_idx}: "
                            f"Input shapes: {[inp.shape for inp in inputs]}, "
                            f"Labels shape: {labels.shape}"
                        )

                        optimizer.zero_grad()
                        feat_m, feat_l, feat_t, outputs = model(inputs)

                        # 计算各模态的单独损失
                        out_m = (
                            torch.mm(feat_m, model.concat_fusion.linear_x_out.weight)
                            + model.concat_fusion.linear_x_out.bias / 2
                        )
                        out_m = (
                            torch.mm(
                                out_m,
                                torch.transpose(model.concat_fusion.clf.weight, 0, 1),
                            )
                            + model.concat_fusion.clf.bias / 2
                        )

                        out_l = (
                            torch.mm(feat_l, model.concat_fusion.linear_y_out.weight)
                            + model.concat_fusion.linear_y_out.bias / 2
                        )
                        out_l = (
                            torch.mm(
                                out_l,
                                torch.transpose(model.concat_fusion.clf.weight, 0, 1),
                            )
                            + model.concat_fusion.clf.bias / 2
                        )

                        out_t = (
                            torch.mm(feat_t, model.concat_fusion.linear_z_out.weight)
                            + model.concat_fusion.linear_z_out.bias / 2
                        )
                        out_t = (
                            torch.mm(
                                out_t,
                                torch.transpose(model.concat_fusion.clf.weight, 0, 1),
                            )
                            + model.concat_fusion.clf.bias / 2
                        )

                        loss = criterion(outputs, labels)
                        loss_m = criterion(out_m, labels)
                        loss_l = criterion(out_l, labels)
                        loss_t = criterion(out_t, labels)

                        # 动态权重计算
                        theta_ratio = self.sigmoid(
                            np.array(
                                [
                                    loss_m.detach().cpu().numpy(),
                                    loss_l.detach().cpu().numpy(),
                                    loss_t.detach().cpu().numpy(),
                                ]
                            )
                        )

                        total_loss = (
                            loss
                            + theta_ratio[0] * loss_m
                            + theta_ratio[1] * loss_l
                            + theta_ratio[2] * loss_t
                        )

                        total_loss.backward()
                        optimizer.step()

                        avg_loss += total_loss.item()
                        train_m_loss += loss_m.item()
                        train_l_loss += loss_l.item()
                        train_t_loss += loss_t.item()

                    except Exception as e:
                        logger.error(f"Error in training batch {batch_idx}: {e}")
                        logger.error(
                            f"Input shapes: {[inp.shape if hasattr(inp, 'shape') else 'No shape' for inp in inputs]}"
                        )
                        raise

                avg_loss /= len(train_loader)
                train_m_loss /= len(train_loader)
                train_l_loss /= len(train_loader)
                train_t_loss /= len(train_loader)

                best_model, eval_m_loss, eval_l_loss, eval_t_loss = (
                    self.evaluate_detailed(model, eval_loader, best_model)
                )

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")

        if best_model[1] is not None:
            model.load_state_dict(best_model[1])

        test_results = self.test(model, test_loader)
        return test_results

    def evaluate_detailed(self, model, eval_loader, best_model):
        """详细的评估函数，返回各模态的损失"""
        model.eval()
        eval_loss = 0
        eval_m_loss = 0
        eval_l_loss = 0
        eval_t_loss = 0

        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs = [_inputs.to(self.device) for _inputs in inputs]
                labels = labels.to(self.device)

                feat_m, feat_l, feat_t, outputs = model(inputs)

                # 计算各模态损失
                out_m = (
                    torch.mm(feat_m, model.concat_fusion.linear_x_out.weight)
                    + model.concat_fusion.linear_x_out.bias / 2
                )
                out_m = (
                    torch.mm(
                        out_m, torch.transpose(model.concat_fusion.clf.weight, 0, 1)
                    )
                    + model.concat_fusion.clf.bias / 2
                )

                out_l = (
                    torch.mm(feat_l, model.concat_fusion.linear_y_out.weight)
                    + model.concat_fusion.linear_y_out.bias / 2
                )
                out_l = (
                    torch.mm(
                        out_l, torch.transpose(model.concat_fusion.clf.weight, 0, 1)
                    )
                    + model.concat_fusion.clf.bias / 2
                )

                out_t = (
                    torch.mm(feat_t, model.concat_fusion.linear_z_out.weight)
                    + model.concat_fusion.linear_z_out.bias / 2
                )
                out_t = (
                    torch.mm(
                        out_t, torch.transpose(model.concat_fusion.clf.weight, 0, 1)
                    )
                    + model.concat_fusion.clf.bias / 2
                )

                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss_m = torch.nn.functional.cross_entropy(out_m, labels)
                loss_l = torch.nn.functional.cross_entropy(out_l, labels)
                loss_t = torch.nn.functional.cross_entropy(out_t, labels)

                eval_loss += loss.item()
                eval_m_loss += loss_m.item()
                eval_l_loss += loss_l.item()
                eval_t_loss += loss_t.item()

        eval_loss /= len(eval_loader)
        eval_m_loss /= len(eval_loader)
        eval_l_loss /= len(eval_loader)
        eval_t_loss /= len(eval_loader)

        if eval_loss <= best_model[0]:
            print(f"Reduce from {best_model[0]:.6f} -> {eval_loss:.6f}")
            best_model = (eval_loss, model.state_dict())

        return best_model, eval_m_loss, eval_l_loss, eval_t_loss

    def test(self, model, test_loader):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                try:
                    logger.debug(
                        f"Testing batch {batch_idx}: "
                        f"Input shapes: {[np.array(inp).shape if not torch.is_tensor(inp) else inp.shape for inp in inputs]}, "
                        f"Labels shape: {np.array(labels).shape if not torch.is_tensor(labels) else labels.shape}"
                    )

                    inputs = [_inputs.to(self.device) for _inputs in inputs]

                    _, _, _, outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

                except Exception as e:
                    logger.error(f"Error in test batch {batch_idx}: {e}")
                    logger.error(f"Input types: {[type(inp) for inp in inputs]}")
                    logger.error(
                        f"Input shapes: {[inp.shape if hasattr(inp, 'shape') else 'No shape' for inp in inputs]}"
                    )
                    raise

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

    def check_data_dimensions(self, description=""):
        """检查所有数据的维度一致性"""
        logger.info(f"=== Dimension Check {description} ===")

        # 检查数据长度
        logger.info(
            f"Data lengths: log={len(self.log_data)}, metric={len(self.metric_data)}, trace={len(self.trace_data)}, labels={len(self.labels)}"
        )

        # 检查每种数据的维度分布
        log_shapes = {}
        metric_shapes = {}
        trace_shapes = {}

        for i, (log, metric, trace) in enumerate(
            zip(self.log_data, self.metric_data, self.trace_data)
        ):
            log_shape = str(np.array(log).shape) if log is not None else "None"
            metric_shape = str(np.array(metric).shape) if metric is not None else "None"
            trace_shape = str(np.array(trace).shape) if trace is not None else "None"

            log_shapes[log_shape] = log_shapes.get(log_shape, []) + [i]
            metric_shapes[metric_shape] = metric_shapes.get(metric_shape, []) + [i]
            trace_shapes[trace_shape] = trace_shapes.get(trace_shape, []) + [i]

        logger.info(
            f"Log shape distribution: {[(shape, len(indices)) for shape, indices in log_shapes.items()]}"
        )
        logger.info(
            f"Metric shape distribution: {[(shape, len(indices)) for shape, indices in metric_shapes.items()]}"
        )
        logger.info(
            f"Trace shape distribution: {[(shape, len(indices)) for shape, indices in trace_shapes.items()]}"
        )

        # 报告异常样本
        if len(trace_shapes) > 1:
            logger.error("CRITICAL: Multiple trace shapes detected!")
            for shape, indices in trace_shapes.items():
                if len(indices) < 5:  # 只显示少数样本的详细信息
                    logger.error(f"  Shape {shape}: samples {indices}")
                else:
                    logger.error(
                        f"  Shape {shape}: {len(indices)} samples (first 5: {indices[:5]})"
                    )

        return len(log_shapes) == 1, len(metric_shapes) == 1, len(trace_shapes) == 1
