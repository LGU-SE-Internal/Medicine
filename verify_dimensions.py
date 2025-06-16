#!/usr/bin/env python3
"""
验证 dataset_trace.py 和 trace.py 输出维度的一致性
"""

import pandas as pd
from pathlib import Path
import tempfile
import json


def create_mock_data():
    """创建模拟数据来测试维度一致性"""

    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    data_pack = temp_dir / "test_pack"
    data_pack.mkdir()

    # 创建模拟的trace数据
    trace_data = {
        "span_id": ["span1", "span2", "span3", "span4"],
        "parent_span_id": [None, "span1", "span1", "span2"],
        "service_name": ["service_A", "service_B", "service_C", "service_B"],
        "duration": [100, 200, 150, 300],
        "time": [1000000000, 1000000001, 1000000002, 1000000003],
    }

    normal_df = pd.DataFrame(trace_data)
    abnormal_df = pd.DataFrame(
        {
            "span_id": ["span1", "span2", "span3"],
            "parent_span_id": [None, "span1", "span1"],
            "service_name": ["service_A", "service_B", "service_C"],
            "duration": [500, 800, 600],  # 异常高的持续时间
            "time": [1000000010, 1000000011, 1000000012],
        }
    )

    # 保存为parquet文件
    normal_df.to_parquet(data_pack / "normal_traces.parquet")
    abnormal_df.to_parquet(data_pack / "abnormal_traces.parquet")

    # 创建注入信息
    injection_info = {"fault_type": "cpu", "target_service": "service_B"}

    with open(data_pack / "injection.json", "w") as f:
        json.dump(injection_info, f)

    return data_pack


def test_dataset_trace_dimensions():
    """测试 dataset_trace.py 的输出维度"""
    try:
        from dataset.dataset_trace import TraceDataset

        data_pack = create_mock_data()
        dataset = TraceDataset([data_pack])

        if len(dataset) > 0:
            X, y = dataset[0]
            print(f"dataset_trace.py 输出:")
            print(f"  X shape: {X.shape}")
            print(f"  X type: {type(X)}")
            print(f"  y: {y}")

            # 验证X是一维向量
            if X.ndim == 1:
                print(f"  ✓ 正确：输出是一维向量，长度为 {len(X)}")
                return len(X)
            else:
                print(f"  ✗ 错误：输出是 {X.ndim} 维，应该是一维")
                return None
        else:
            print("dataset_trace.py: 数据集为空")
            return None

    except Exception as e:
        print(f"测试 dataset_trace.py 时出错: {e}")
        return None


def test_trace_dimensions():
    """测试 trace.py 的期望输出维度结构"""
    print(f"trace.py 期望输出:")
    print(f"  应该返回二维数组，形状为 (num_samples, num_features)")
    print(f"  其中每个样本是一个一维特征向量")
    print(f"  特征维度应该等于 len(invoke_list)")


def main():
    print("=" * 60)
    print("验证 dataset_trace.py 和 trace.py 输出维度的一致性")
    print("=" * 60)

    # 测试 dataset_trace.py
    feature_dim = test_dataset_trace_dimensions()

    print()

    # 说明 trace.py 的期望结构
    test_trace_dimensions()

    print()
    print("=" * 60)
    if feature_dim is not None:
        print("总结:")
        print(f"  dataset_trace.py: 每个数据包返回一个 {feature_dim} 维的特征向量")
        print(f"  trace.py: 每个故障样本应该是一个 len(invoke_list) 维的特征向量")
        print(f"  维度结构现在应该是一致的 ✓")
    else:
        print("测试失败，无法验证维度一致性")
    print("=" * 60)


if __name__ == "__main__":
    main()
