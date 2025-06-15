import pickle
import torch
import os
import numpy as np
import random
import pandas as pd
import logging
import json


def notice(msg):
    print(f"\033[35m{msg}\033[0m")


def set_seed(seed=2024):
    torch.random.manual_seed(2024)
    np.random.seed(2024)
    random.seed(seed)
    print(f"Seed: {seed}")


def check(save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def dump(obj, save_path):
    check(save_path)
    with open(save_path, "wb") as w:
        pickle.dump(obj, w)


def get_files(root: str, keyword: str):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.find(keyword) != -1:
                files.append(os.path.join(dirpath, filename))

    return files


def save_result(labels, preds, save_path):
    check(save_path)
    pd.DataFrame({"label": labels, "pred": preds}).sort_values(by="label").to_csv(
        save_path
    )


def set_logger(config, time):
    logger = logging.getLogger(config["log_name"])
    logger.setLevel(logging.INFO)
    log_dir = os.path.join("result", config["dataset"], "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(os.path.join(log_dir, time + ".log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.info("\n[Experiment config]")
    for key, value in config.items():
        logger.info(f"-> {key:20}: {str(value)}")


def get_device(config):
    if config["use_cuda"]:
        # os.environ['CUDA_VISIBLE_DEVICES']='0'
        device = torch.device(
            f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device("cpu")
    return device


def load_injection_data(injection_file_path: str) -> tuple[str, str]:
    """Load injection data and extract fault_type and target_service.

    Args:
        injection_file_path (str): Path to the injection JSON file.

    Returns:
        tuple[str, str]: A tuple of (fault_type, target_service).

    Raises:
        KeyError: If required keys are missing from the injection data.
    """
    with open(injection_file_path, "r") as f:
        injection = json.load(f)

        # 检查 injection 字典中的必要键
        if "fault_type" not in injection:
            raise KeyError(
                f"'fault_type' key not found in injection: {list(injection.keys())}"
            )
        if "display_config" not in injection:
            raise KeyError(
                f"'display_config' key not found in injection: {list(injection.keys())}"
            )

        fault_type = injection["fault_type"]
        engine = json.loads(injection["display_config"])

        # 检查 engine 字典中的必要键
        if "injection_point" not in engine:
            raise KeyError(
                f"'injection_point' key not found in engine: {list(engine.keys())}"
            )

        target_service = None
        if "app_name" in engine["injection_point"]:
            target_service = engine["injection_point"]["app_name"]
        elif "source_service" in engine["injection_point"]:
            target_service = engine["injection_point"]["source_service"]

        if target_service is None:
            target_service = "unknown_service"

    return fault_type, target_service
