from sklearn.model_selection import train_test_split
import torch
from typing import Optional, Literal
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
import wandb
import os


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
        use_wandb: bool = True,
        wandb_project: str = "multimodal-medicine",
        wandb_name: Optional[str] = None,
    ) -> None:
        # 配置logger
        logger.add("debug.log", rotation="500 MB", level="DEBUG")
        logger.info("Initializing MultiModalTrainer")

        self.data_paths = data_paths
        self.cache_dir = cache_dir
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb

        # 初始化wandb
        if self.use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    name=wandb_name,
                    config=self.config,
                    tags=["multimodal", "medicine", "fault-diagnosis"],
                )
                logger.info("WandB initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.use_wandb = False

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

        # 添加数据增强
        train_data = self.data_enhance(train_data)

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

        # 统一时序维度 - 找到最大时间步长度
        max_time_steps_log = (
            max([shape[0] for shape in log_shapes]) if log_shapes else 0
        )
        max_time_steps_metric = (
            max([shape[1] for shape in metric_shapes]) if metric_shapes else 0
        )

        max_time_steps = max(max_time_steps_log, max_time_steps_metric)

        logger.info(f"Padding to unified time steps: {max_time_steps}")

        if len(set(log_shapes)) > 1:
            logger.warning(
                "Log data dimension mismatch detected - applying padding/truncation"
            )
            log_batch = self._pad_or_truncate_log_sequences(log_batch, max_time_steps)

        if len(set(metric_shapes)) > 1:
            logger.warning(
                "Metric data dimension mismatch detected - applying padding/truncation"
            )
            metric_batch = self._pad_or_truncate_metric_sequences(
                metric_batch, max_time_steps
            )

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
            logger.error("Attempting to create tensors from processed data...")

            processed_log_shapes = [
                np.array(log).shape for log in log_batch if log is not None
            ]
            processed_metric_shapes = [
                np.array(metric).shape for metric in metric_batch if metric is not None
            ]
            processed_trace_shapes = [
                np.array(trace).shape for trace in trace_batch if trace is not None
            ]

            logger.error(f"  Processed log_batch shapes: {processed_log_shapes}")
            logger.error(f"  Processed metric_batch shapes: {processed_metric_shapes}")
            logger.error(f"  Processed trace_batch shapes: {processed_trace_shapes}")

            raise

    def sigmoid(self, data):
        fz = []
        for num in data:
            fz.append(1 / (1 + math.exp(-num)))
        return fz

    def train(self, epochs: int = 100, batch_size: int = 32, lr: float = 1e-3):
        train_loader, eval_loader, test_loader = self.get_data_loaders(batch_size)

        model = self.create_model()
        model.to(self.device)

        optim_type = self.config.get("optim", "AdamW")
        if optim_type == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=self.config.get("weight_decay", 1e-4),
            )
        elif optim_type == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=self.config.get("weight_decay", 1e-4),
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=self.config.get("weight_decay", 1e-4),
            )

        criterion = torch.nn.CrossEntropyLoss()
        best_model = (torch.inf, None)

        alpha = 0.5
        beta = 0.3
        st_time = time.time()

        if self.use_wandb:
            wandb.watch(model, log="all", log_freq=10)
            wandb.config.update(
                {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "optimizer": optim_type,
                    "num_classes": self.num_classes,
                    "device": str(self.device),
                    "log_feature_dim": self.log_feature_dim,
                    "metric_feature_dim": self.metric_feature_dim,
                    "trace_feature_dim": self.trace_feature_dim,
                    "instance_num": self.instance_num,
                    "alpha": alpha,
                    "beta": beta,
                }
            )

        print("Starting training...")
        for epoch in tqdm(range(epochs), desc="Training"):
            model.train()
            if epoch < 1 or epoch > 80:
                avg_loss = 0
                train_m_loss = 0
                train_l_loss = 0
                train_t_loss = 0

                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    try:
                        inputs = [_inputs.to(self.device) for _inputs in inputs]
                        labels = labels.to(self.device)

                        optimizer.zero_grad()
                        feat_m, feat_l, feat_t, outputs = model(inputs)

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
                        raise

                avg_loss /= len(train_loader)
                train_m_loss /= len(train_loader)
                train_l_loss /= len(train_loader)
                train_t_loss /= len(train_loader)

                logger.info(f"epoch: {epoch: 3}, training loss: {avg_loss}")
                best_model, eval_m_loss, eval_l_loss, eval_t_loss = (
                    self.evaluate_with_modals(model, eval_loader, best_model)
                )

                if self.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train/total_loss": avg_loss,
                            "train/metric_loss": train_m_loss,
                            "train/log_loss": train_l_loss,
                            "train/trace_loss": train_t_loss,
                            "eval/metric_loss": eval_m_loss,
                            "eval/log_loss": eval_l_loss,
                            "eval/trace_loss": eval_t_loss,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        step=epoch,
                    )
            else:
                # 梯度优化阶段 (epoch 1-80)
                avg_loss = 0
                cur_m_loss = 0
                cur_l_loss = 0
                cur_t_loss = 0

                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    try:
                        inputs = [_inputs.to(self.device) for _inputs in inputs]
                        labels = labels.to(self.device)

                        optimizer.zero_grad()
                        feat_m, feat_l, feat_t, outputs = model(inputs)

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

                        total_loss = loss + loss_m + loss_l + loss_t
                        total_loss.backward()

                        avg_loss += total_loss.item()
                        cur_m_loss += loss_m.item()
                        cur_l_loss += loss_l.item()
                        cur_t_loss += loss_t.item()

                    except Exception as e:
                        logger.error(f"Error in training batch {batch_idx}: {e}")
                        raise

                avg_loss /= len(train_loader)
                cur_m_loss /= len(train_loader)
                cur_l_loss /= len(train_loader)
                cur_t_loss /= len(train_loader)

                logger.info(f"epoch: {epoch: 3}, training loss: {avg_loss}")
                best_model, cur_eval_m_loss, cur_eval_l_loss, cur_eval_t_loss = (
                    self.evaluate_with_modals(model, eval_loader, best_model)
                )

                minus_eval_m = (
                    (eval_m_loss - cur_eval_m_loss)
                    if eval_m_loss > cur_eval_m_loss
                    else 0.0001
                )
                minus_train_m = (
                    (train_m_loss - cur_m_loss) if train_m_loss > cur_m_loss else 0.0001
                )
                minus_eval_l = (
                    (eval_l_loss - cur_eval_l_loss)
                    if eval_l_loss > cur_eval_l_loss
                    else 0.0001
                )
                minus_train_l = (
                    (train_l_loss - cur_l_loss) if train_l_loss > cur_l_loss else 0.0001
                )
                minus_eval_t = (
                    (eval_t_loss - cur_eval_t_loss)
                    if eval_t_loss > cur_eval_t_loss
                    else 0.0001
                )
                minus_train_t = (
                    (train_t_loss - cur_t_loss) if train_t_loss > cur_t_loss else 0.0001
                )

                ratio_m = minus_eval_m / minus_train_m
                ratio_l = minus_eval_l / minus_train_l
                ratio_t = minus_eval_t / minus_train_t

                theta_ratio = self.sigmoid(np.array([ratio_m, ratio_l, ratio_t]))
                value_ratio, index_ratio = torch.sort(torch.tensor(theta_ratio))
                coeffs = [1.0, 1.0, 1.0]
                coeffs[int(index_ratio[0].item())] = 1 - alpha * float(
                    value_ratio[0].item()
                )
                coeffs[int(index_ratio[-1].item())] = 1 + beta * float(
                    value_ratio[-1].item()
                )

                for name, parms in model.named_parameters():
                    if parms.grad is not None:
                        layer = str(name).split(".")[0]
                        if "metric" in layer:
                            parms.grad = parms.grad * coeffs[0] + torch.zeros_like(
                                parms.grad
                            ).normal_(0, parms.grad.std().item() + 1e-8)
                        elif "log" in layer:
                            parms.grad = parms.grad * coeffs[1] + torch.zeros_like(
                                parms.grad
                            ).normal_(0, parms.grad.std().item() + 1e-8)
                        elif "trace" in layer:
                            parms.grad = parms.grad * coeffs[2] + torch.zeros_like(
                                parms.grad
                            ).normal_(0, parms.grad.std().item() + 1e-8)

                optimizer.step()

                train_m_loss = cur_m_loss
                train_l_loss = cur_l_loss
                train_t_loss = cur_t_loss
                eval_m_loss = cur_eval_m_loss
                eval_l_loss = cur_eval_l_loss
                eval_t_loss = cur_eval_t_loss

                if self.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train/total_loss": avg_loss,
                            "train/metric_loss": cur_m_loss,
                            "train/log_loss": cur_l_loss,
                            "train/trace_loss": cur_t_loss,
                            "eval/metric_loss": cur_eval_m_loss,
                            "eval/log_loss": cur_eval_l_loss,
                            "eval/trace_loss": cur_eval_t_loss,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "gradient_opt/ratio_m": ratio_m,
                            "gradient_opt/ratio_l": ratio_l,
                            "gradient_opt/ratio_t": ratio_t,
                            "gradient_opt/coeff_metric": coeffs[0],
                            "gradient_opt/coeff_log": coeffs[1],
                            "gradient_opt/coeff_trace": coeffs[2],
                            "gradient_opt/alpha": alpha,
                            "gradient_opt/beta": beta,
                        },
                        step=epoch,
                    )

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")

        training_time = time.time() - st_time
        logger.info(f"Training time={training_time: .6}, #Case={len(train_loader)}")

        if best_model[1] is not None:
            model.load_state_dict(best_model[1])

        test_results = self.test(model, test_loader)

        if self.use_wandb:
            wandb.log(
                {
                    "final/training_time": training_time,
                    "final/best_eval_loss": best_model[0],
                    "final/test_precision_macro": test_results[2]["macro"][0],
                    "final/test_recall_macro": test_results[2]["macro"][1],
                    "final/test_f1_macro": test_results[2]["macro"][2],
                    "final/test_precision_weighted": test_results[2]["weighted"][0],
                    "final/test_recall_weighted": test_results[2]["weighted"][1],
                    "final/test_f1_weighted": test_results[2]["weighted"][2],
                }
            )

            try:
                from sklearn.metrics import confusion_matrix
                import matplotlib.pyplot as plt
                import seaborn as sns

                cm = confusion_matrix(test_results[1], test_results[0])
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title("Confusion Matrix")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                wandb.log({"confusion_matrix": wandb.Image(plt)})
                plt.close()
            except Exception as e:
                logger.warning(f"Failed to create confusion matrix: {e}")

            wandb.finish()

        return test_results

    def experiment(self):
        self.experiment_train("all")
        self.experiment_train("log")
        self.experiment_train("metric")
        self.experiment_train("trace")

    def experiment_train(self, modal: Literal["all", "log", "metric", "trace"]):
        logger.info(f"Single modal training [{modal}]")
        print(f"Single modal training [{modal}]")

        if self.use_wandb:
            wandb.init(
                project="multimodal-medicine-single",
                name=f"single_{modal}",
                config={**self.config, "modal": modal},
                tags=["single-modal", modal, "experiment"],
                reinit=True,
            )

        model = self.create_experiment_model()
        train_loader, eval_loader, test_loader = self.get_data_loaders()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get("lr", 1e-3),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

        model.set_use_modal(modal)
        model.to(self.device)

        best_model = (torch.inf, None)
        criterion = torch.nn.CrossEntropyLoss()
        record_loss = []

        epochs = self.config.get("epochs", 100)

        for epoch in tqdm(range(epochs), desc="Training"):
            model.train()
            avg_loss = 0

            for inputs, labels in train_loader:
                inputs = [_inputs.to(self.device) for _inputs in inputs]
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()

            avg_loss /= len(train_loader)
            record_loss.append(avg_loss)
            logger.info(f"epoch: {epoch: 3}, training loss: {avg_loss}")
            best_model = self.experiment_eval(model, eval_loader, best_model, modal)

            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        f"train_{modal}/loss": avg_loss,
                        f"eval_{modal}/loss": best_model[0]
                        if best_model[0] != torch.inf
                        else avg_loss,
                        f"train_{modal}/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )

        import json

        result_dir = os.path.join("result", self.config.get("dataset", "default"))
        os.makedirs(result_dir, exist_ok=True)

        with open(
            os.path.join(result_dir, f"single_train_{modal}_loss.json"),
            "w",
            encoding="utf8",
        ) as w:
            json.dump(
                {
                    "loss": record_loss,
                    "epoch": list(range(1, epochs + 1)),
                },
                w,
            )

        if best_model[1] is not None:
            model.load_state_dict(best_model[1])

        test_results = self.experiment_test(model, test_loader, modal)

        if self.use_wandb:
            wandb.log(
                {
                    f"final_{modal}/test_precision_macro": test_results[2]["macro"][0],
                    f"final_{modal}/test_recall_macro": test_results[2]["macro"][1],
                    f"final_{modal}/test_f1_macro": test_results[2]["macro"][2],
                    f"final_{modal}/test_precision_weighted": test_results[2][
                        "weighted"
                    ][0],
                    f"final_{modal}/test_recall_weighted": test_results[2]["weighted"][
                        1
                    ],
                    f"final_{modal}/test_f1_weighted": test_results[2]["weighted"][2],
                }
            )
            wandb.finish()

        return test_results

    def experiment_eval(
        self,
        model: ExperimentModel,
        eval_loader,
        best_model,
        modal: Literal["all", "log", "metric", "trace"],
    ):
        model.set_use_modal(modal)
        model.eval()
        eval_loss = 0

        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs = [_inputs.to(self.device) for _inputs in inputs]
                labels = labels.to(self.device)
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                eval_loss += loss.item()

        eval_loss /= len(eval_loader)
        if eval_loss <= best_model[0]:
            logger.info(f"Reduce from {best_model[0]: .6f} -> {eval_loss: .6f}")
            best_model = (eval_loss, model.state_dict())

        return best_model

    def experiment_test(
        self,
        model: ExperimentModel,
        test_loader,
        modal: Literal["all", "log", "metric", "trace"],
    ):
        logger.info(f"Testing under {modal}")
        model.set_use_modal(modal)
        model.eval()

        all_labels = []
        all_preds = []
        test_loss = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = [_inputs.to(self.device) for _inputs in inputs]
                labels = labels.to(self.device)
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                test_loss += loss.item()
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(outputs.argmax(dim=1).cpu().tolist())

        test_loss /= len(test_loader)
        logger.info(f"Testing loss={test_loss: .6}")
        print(f"Testing loss={test_loss: .6}")

        return (
            all_preds,
            all_labels,
            {
                "macro": self.calculate_metrics_detailed(
                    all_labels, all_preds, "macro"
                ),
                "weighted": self.calculate_metrics_detailed(
                    all_labels, all_preds, "weighted"
                ),
            },
        )

    def calculate_metrics_detailed(
        self,
        y_true,
        y_pred,
        method: Literal["micro", "macro", "samples", "weighted"] = "weighted",
        to_stdout=True,
        to_logger=True,
    ):
        precision = precision_score(
            y_true,
            y_pred,
            average=method,
            zero_division=0,
        )
        recall = recall_score(y_true, y_pred, average=method, zero_division=0)
        f1score = f1_score(y_true, y_pred, average=method)

        if to_logger:
            logger.info(f"{method} precision:{precision}")
            logger.info(f"{method} recall   :{recall}")
            logger.info(f"{method} f1score  :{f1score}")

        if to_stdout:
            print(f"{method} precision:{precision}")
            print(f"{method} recall   :{recall}")
            print(f"{method} f1score  :{f1score}")

        return [precision, recall, f1score]

    def test(self, model, test_loader):
        model.eval()
        all_preds = []
        all_labels = []
        test_loss = 0
        st_time = time.time()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                try:
                    inputs = [_inputs.to(self.device) for _inputs in inputs]
                    labels = labels.to(self.device)

                    _, _, _, outputs = model(inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    test_loss += loss.item()
                    all_labels.extend(labels.cpu().tolist())
                    all_preds.extend(outputs.argmax(dim=1).cpu().tolist())

                except Exception as e:
                    logger.error(f"Error in test batch {batch_idx}: {e}")
                    raise

        testing_time = time.time() - st_time
        logger.info(f"Testing time={testing_time: .6}, #Case={len(test_loader)}")
        test_loss /= len(test_loader)
        logger.info(f"Testing loss={test_loss: .6}")
        print(f"Testing loss={test_loss: .6}")

        return (
            all_preds,
            all_labels,
            {
                "macro": self.calculate_metrics_detailed(
                    all_labels, all_preds, "macro"
                ),
                "weighted": self.calculate_metrics_detailed(
                    all_labels, all_preds, "weighted"
                ),
            },
        )

    def calculate_metrics(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        return {"precision": precision, "recall": recall, "f1": f1}

    def check_data_dimensions(self, description=""):
        logger.info(f"=== Dimension Check {description} ===")

        logger.info(
            f"Data lengths: log={len(self.log_data)}, metric={len(self.metric_data)}, trace={len(self.trace_data)}, labels={len(self.labels)}"
        )

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

        if len(trace_shapes) > 1:
            logger.error("CRITICAL: Multiple trace shapes detected!")
            for shape, indices in trace_shapes.items():
                if len(indices) < 5:
                    logger.error(f"  Shape {shape}: samples {indices}")
                else:
                    logger.error(
                        f"  Shape {shape}: {len(indices)} samples (first 5: {indices[:5]})"
                    )

        return len(log_shapes) == 1, len(metric_shapes) == 1, len(trace_shapes) == 1

    def _pad_or_truncate_log_sequences(self, log_batch, target_length):
        processed_batch = []

        for log_seq in log_batch:
            log_array = np.array(log_seq)
            current_length = log_array.shape[0]
            feature_dim = log_array.shape[1]

            if current_length == target_length:
                processed_batch.append(log_seq)
            elif current_length < target_length:
                padding_length = target_length - current_length
                padding = np.zeros((padding_length, feature_dim))
                padded_seq = np.concatenate([log_array, padding], axis=0)
                processed_batch.append(padded_seq.tolist())
                logger.debug(
                    f"Padded log sequence from {current_length} to {target_length}"
                )
            else:
                truncated_seq = log_array[:target_length]
                processed_batch.append(truncated_seq.tolist())
                logger.debug(
                    f"Truncated log sequence from {current_length} to {target_length}"
                )

        return processed_batch

    def _pad_or_truncate_metric_sequences(self, metric_batch, target_length):
        processed_batch = []

        for metric_seq in metric_batch:
            metric_array = np.array(metric_seq)
            # metric shape: (services, time_steps, features)
            num_services = metric_array.shape[0]
            current_length = metric_array.shape[1]
            feature_dim = metric_array.shape[2]

            if current_length == target_length:
                # No change needed
                processed_batch.append(metric_seq)
            elif current_length < target_length:
                # Pad with zeros
                padding_length = target_length - current_length
                padding = np.zeros((num_services, padding_length, feature_dim))
                padded_seq = np.concatenate([metric_array, padding], axis=1)
                processed_batch.append(padded_seq.tolist())
                logger.debug(
                    f"Padded metric sequence from {current_length} to {target_length}"
                )
            else:
                # Truncate
                truncated_seq = metric_array[:, :target_length, :]
                processed_batch.append(truncated_seq.tolist())
                logger.debug(
                    f"Truncated metric sequence from {current_length} to {target_length}"
                )

        return processed_batch

    def data_enhance(self, data):
        y_dict = {}
        for index, sample in enumerate(data):
            if y_dict.get(sample[3], None) is None:
                y_dict[sample[3]] = []
            y_dict[sample[3]].append(index)

        enhance_num = max([len(val) for val in y_dict.values()])
        scheduler = tqdm(total=enhance_num * len(y_dict.keys()), desc="Data enhancing")
        new_data = []

        for _, indices in y_dict.items():
            cnt = len(indices)
            scheduler.update(cnt)
            while cnt < enhance_num:
                new_data.append(self.fake_sample(indices, data))
                cnt += 1
                scheduler.update(1)

        scheduler.close()
        data.extend(new_data)
        return data

    def fake_sample(self, indices, data):
        choices = random.choices(indices, k=2)
        sample1 = data[choices[0]]
        sample2 = data[choices[1]]

        return (
            random.choice([sample1[0], sample2[0]]),  # log
            random.choice([sample1[1], sample2[1]]),  # metric
            random.choice([sample1[2], sample2[2]]),  # trace
            sample1[3],
        )

    def evaluate_with_modals(self, model, eval_loader, best_model):
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
            logger.info(f"Reduce from {best_model[0]:.6f} -> {eval_loss:.6f}")
            best_model = (eval_loss, model.state_dict())

        return best_model, eval_m_loss, eval_l_loss, eval_t_loss
