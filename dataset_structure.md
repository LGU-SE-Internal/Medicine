# Dataset 结构说明

本文档说明了 Medicine 框架中数据集的结构和特征输出格式，为在新数据上构建数据集提供参考。

## 1. 整体架构

该数据集系统支持三种模态的数据：
- **Log 数据集** (日志数据)
- **Metric 数据集** (指标数据) 
- **Trace 数据集** (链路追踪数据)

所有数据集都继承自 `BaseDataset` 基类，提供统一的接口和数据格式。

## 2. 基础数据结构

### 2.1 BaseDataset 核心属性

```python
class BaseDataset:
    def __init__(self, config, modal):
        # 数据存储
        self.__X__ = []              # 特征数据列表 - 每个元素是一个样本的特征
        self.__y__ = {               # 标签数据字典 - 每个列表与__X__一一对应
            "failure_type": [],      # 故障类型标签列表
            "root_cause": []         # 根因标签列表  
        }
        
        # 配置信息
        self.services = []           # 服务列表
        self.instances = []          # 实例列表  
        self.failures = []           # 故障类型列表
```

**重要**: `__X__[i]` 和 `__y__["failure_type"][i]`、`__y__["root_cause"][i]` 是一一对应的，即第i个样本的特征对应第i个标签。

### 2.2 数据访问接口

```python
@property
def X(self):
    return self.__X__              # 返回特征数据列表

@property  
def y(self):
    return self.__y__[self.__config__["label_type"]]  # 根据配置返回对应标签列表

def __getitem__(self, index):
    # 返回第index个样本的特征和标签
    return self.__X__[index], self.__y__[self.__config__["label_type"]][index]

def __len__(self):
    return len(self.__X__)         # 返回样本总数
```

**数据对应关系**:
- `len(self.__X__) == len(self.__y__["failure_type"]) == len(self.__y__["root_cause"])`
- 每个索引i对应一个完整的样本: 特征`__X__[i]` + 标签`__y__["failure_type"][i]` + `__y__["root_cause"][i]`

## 2.5 数据样本对应关系

数据集中的特征和标签严格按照索引一一对应：

```python
# 示例：假设有3个样本
__X__ = [
    sample_0_features,    # 第0个样本的特征
    sample_1_features,    # 第1个样本的特征  
    sample_2_features     # 第2个样本的特征
]

__y__ = {
    "failure_type": [0, 1, 2],    # 对应每个样本的故障类型标签
    "root_cause": [0, 1, 0]       # 对应每个样本的根因标签
}
```

**添加样本时的同步操作**:
```python
# 在LogDataset中的示例
def __add_sample__(self, st_time, cnts, log_df, label):
    # ... 处理特征数据 ...
    self.__X__.append(new_seq)                              # 添加特征
    self.__y__["failure_type"].append(label["failure_type"]) # 添加对应的故障类型标签
    self.__y__["root_cause"].append(label["root_cause"])     # 添加对应的根因标签
```

## 3. 配置文件结构

每个数据集需要的核心配置参数：

```python
config = {
    # 数据集基本信息
    "dataset": "数据集名称",
    "dataset_dir": "数据集目录路径", 
    "save_dir": "保存目录",
    "sample_interval": 60,          # 采样间隔(秒)
    "num_workers": 10,              # 并行处理数
    
    # 标签映射信息
    "failures": "cpu memory network io process",     # 故障类型列表(空格分隔)
    "services": "service1 service2 service3",        # 服务列表(空格分隔)
    "instances": "instance1 instance2 instance3",    # 实例列表(空格分隔)
    "label_type": "failure_type",   # 使用的标签类型
    "num_class": 5,                 # 分类数量
    
    # 模态特定配置
    "drain_config": {...},          # Log模态的Drain配置
    "bert_config": {...},           # Log模态的BERT配置
}
```

## 4. 各模态数据结构

### 4.1 Log 数据集

#### 输入数据格式
- **日志数据**: `[timestamp, message]`
- **标注数据**: `[st_time, ed_time, failure_type, root_cause]`

#### 输出特征格式
```python
X: List[List[List[float]]]  # [样本数, 时间序列长度, 768维BERT嵌入]
```

- **维度**: `[sample_num, sequence_length, 768]`
- **特征描述**: 每个样本是一个时间序列，每个时间步包含该时间窗口内日志模板的加权BERT嵌入向量
- **处理流程**:
  1. 使用Drain算法从原始日志提取模板
  2. 使用BERT对模板进行编码(768维)
  3. 根据模板的异常权重进行加权求和

#### 输出标签格式
```python
y: {
    "failure_type": List[int],    # [样本数] - 故障类型分类标签(0到num_class-1)
    "root_cause": List[int]       # [样本数] - 根因服务标签(服务索引)
}
```

- **failure_type**: 整数标签，表示故障类型分类（如：0=cpu, 1=memory, 2=network等）
- **root_cause**: 整数标签，表示根因服务索引（对应配置中的services列表索引）
- **标签编码**: 通过`self.failures.index()`和`self.services.index()`将字符串转为索引

#### 核心组件
- **DrainProcesser**: 日志模板提取
- **BertEncoder**: BERT编码器，输出768维向量

### 4.2 Metric 数据集

#### 输入数据格式
- **指标数据**: `[timestamp, cmdb_id, kpi_name, value]`
- **标注数据**: `[st_time, ed_time, failure_type, root_cause]`

#### 输出特征格式
```python
X: List[List[List[List[float]]]]  # [样本数, 实例数, 时间序列长度, KPI数量]
```

- **维度**: `[sample_num, instance_num, sequence_length, kpi_num]`
- **特征描述**: 每个样本包含多个实例，每个实例有多个时间步，每个时间步包含所有KPI指标值
- **处理流程**:
  1. 按实例和KPI分组
  2. 对时间序列进行标准化和差分处理
  3. 按时间窗口采样

#### 输出标签格式
```python
y: {
    "failure_type": List[int],    # [样本数] - 故障类型分类标签(0到num_class-1)
    "root_cause": List[int]       # [样本数] - 根因服务标签(服务索引)
}
```

- **failure_type**: 整数标签，表示故障类型分类（如：0=cpu, 1=memory, 2=network等）
- **root_cause**: 整数标签，表示根因服务索引（对应配置中的services列表索引）
- **标签编码**: 通过`self.failures.index()`和`self.services.index()`将字符串转为索引

#### 特征维度方法
```python
def get_feature(self):
    return self.kpi_num  # 返回KPI指标数量
```

### 4.3 Trace 数据集

#### 输入数据格式
- **链路数据**: `[timestamp, cmdb_id, ccmdb_id, duration, invoke_link, ...]`
- **标注数据**: `[st_time, ed_time, failure_type, root_cause]`

#### 输出特征格式
```python
X: List[List[float]]  # [样本数, 调用链数量]
```

- **维度**: `[sample_num, invoke_link_num]`
- **特征描述**: 每个样本是一个向量，每个维度代表一个调用链的异常程度
- **处理流程**:
  1. 提取服务间调用关系(`invoke_link = service_a + "_" + service_b`)
  2. 计算每个调用链的异常分数(基于Z-score)
  3. 根据调用链的重要性权重加权

#### 输出标签格式
```python
y: {
    "failure_type": List[int],    # [样本数] - 故障类型分类标签(0到num_class-1)
    "root_cause": List[int]       # [样本数] - 根因服务标签(服务索引)
}
```

- **failure_type**: 整数标签，表示故障类型分类（如：0=cpu, 1=memory, 2=network等）
- **root_cause**: 整数标签，表示根因服务索引（对应配置中的services列表索引）
- **标签编码**: 通过`self.failures.index()`和`self.services.index()`将字符串转为索引

#### 拓扑信息
```python
self.topo = [[], []]  # [源节点列表, 目标节点列表]，用于构建服务拓扑图
```

#### 特征维度方法
```python
def get_feature(self):
    return len(self.invoke_list)  # 返回调用链数量
```

### 4.4 所有模态的统一标签格式

**重要说明**: 无论是Log、Metric还是Trace模态，所有数据集的标签格式都是统一的：

```python
y: {
    "failure_type": List[int],    # 故障类型标签列表
    "root_cause": List[int]       # 根因标签列表
}
```

### 标签值范围
- **failure_type**: `[0, num_class-1]` - 故障类型的分类索引
- **root_cause**: `[0, len(services)-1]` - 服务列表中的索引

### 具体示例
```python
# 配置示例
config = {
    "failures": "cpu memory network io process",    # 5个故障类型
    "services": "service1 service2 service3",       # 3个服务
    "num_class": 5
}

# 对应的标签值
failure_type_labels = [0, 1, 2, 3, 4]  # cpu=0, memory=1, network=2, io=3, process=4
root_cause_labels = [0, 1, 2]          # service1=0, service2=1, service3=2
```

## 5. 标签处理

### 5.1 故障类型映射
每个数据集都定义了 `ANOMALY_DICT` 将原始故障类型映射到标准类型：

```python
ANOMALY_DICT = {
    "cpu anomaly": "cpu",
    "memory overload": "memory", 
    "network delay": "network",
    "pod anomaly": "pod_failure",
    # ...
}
```

### 5.2 标签编码
- **failure_type**: 故障类型标签，转换为分类索引
- **root_cause**: 根因标签，转换为服务/实例索引

```python
# 标签编码示例
failure_type_index = self.failures.index(case["failure_type"])
root_cause_index = self.services.index(case["root_cause"])
```

## 6. 数据预处理

### 6.1 时间窗口处理
所有数据集都基于时间窗口进行处理：
- 根据 `sample_interval` 将时间划分为固定窗口
- 每个异常事件扩展为前后各600秒的时间范围
- 按时间窗口聚合和采样数据

### 6.2 数据持久化
```python
# 数据保存格式
{
    "X": self.__X__,           # 特征数据
    "y": self.__y__,           # 标签数据  
    "appendix": {...}          # 模态特定的附加信息
}
```

## 7. 实现新数据集的要点

### 7.1 必需实现的方法
```python
class YourDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config, "your_modal")
        self.ANOMALY_DICT = {...}  # 定义故障类型映射
        
    def load(self):
        # 实现数据加载逻辑
        pass
        
    def get_feature(self):  # 对于metric和trace模态
        # 返回特征维度
        pass
```

### 7.2 数据格式要求
- **特征数据** (`X`): 根据模态类型决定维度结构
- **标签数据** (`y`): 必须包含 `failure_type` 和 `root_cause`
- **配置信息**: 必须包含服务列表、故障类型列表等映射信息

### 7.3 关键注意事项
1. 确保时间戳格式统一(Unix时间戳)
2. 标签映射必须与配置文件中的列表一致
3. 特征维度要与模型输入要求匹配
4. 实现适当的数据预处理和归一化

## 8. 支持的数据集

当前框架支持的数据集：
- **AIOps22**: 企业级微服务数据
- **Platform**: 云平台监控数据  
- **Gaia**: 开源微服务数据

每个数据集都有对应的Log、Metric、Trace三个版本的实现。

## 9. Trainer的数据集加载方式

### 9.1 分模态加载机制

Trainer采用**分模态加载**的方式，分别实例化三个数据集：

```python
class MultiModalTrainer:
    def __init__(self, config, time):
        # 分别加载三种模态的数据集
        self.__log_data__ = self.__load_dataset__(
            LOG_DATASET[config["dataset"]](config)
        )
        self.__metric_data__ = self.__load_dataset__(
            METRIC_DATASET[config["dataset"]](config)
        )
        self.__trace_data__ = self.__load_dataset__(
            TRACE_DATASET[config["dataset"]](config)
        )
```

### 9.2 数据集注册机制

数据集通过字典方式注册在`dataset/__init__.py`中：

```python
# dataset/__init__.py
LOG_DATASET = {
    "aiops22": Aiops22Log,
    "platform": PlatformLog, 
    "gaia": GaiaLog
}
METRIC_DATASET = {
    "aiops22": Aiops22Metric,
    "platform": PlatformMetric,
    "gaia": GaiaMetric
}
TRACE_DATASET = {
    "aiops22": Aiops22Trace,
    "platform": PlatformTrace,
    "gaia": GaiaTrace
}
```

### 9.3 数据集缓存机制

```python
def __load_dataset__(self, dataset):
    if self.__config__["use_tmp"] and os.path.exists(dataset.get_dataset_path()):
        print("Use: cached dataset")
        dataset.load_from_tmp()        # 从缓存文件加载
    else:
        dataset.load()                 # 重新处理原始数据
        dataset.save_to_tmp()          # 保存到缓存文件
    return dataset
```

### 9.4 多模态数据组合

在DataLoader中，trainer将三种模态的数据组合成统一的训练样本：

```python
def __get_loader__(self):
    # 将三种模态的数据打包成元组列表
    data = list(zip(
        self.__log_data__.X,      # Log特征
        self.__metric_data__.X,   # Metric特征  
        self.__trace_data__.X,    # Trace特征
        self.__log_data__.y,      # 标签(三种模态共享相同标签)
    ))
    
    # 数据增强后返回DataLoader
    train_data = []
    for index in X_train:
        train_data.append(data[index])
    return DataLoader(train_data, ...)
```

### 9.5 关键约束条件

1. **样本数量一致**: 三种模态的数据集必须有相同的样本数量
   ```python
   len(log_data.X) == len(metric_data.X) == len(trace_data.X)
   ```

2. **标签一致**: 所有模态使用相同的标签，trainer只使用log_data的标签
   ```python
   # 只使用log_data的标签，其他模态的标签应该相同
   self.__log_data__.y
   ```

3. **索引对应**: 第i个样本在三种模态中必须对应同一个故障事件

### 9.6 实现统一数据集的可能性

**回答您的问题：是否可以实现一个统一的dataset**

理论上**可以**实现一个统一的数据集类，但需要考虑以下几点：

#### 优势：
- 数据一致性更容易保证
- 避免三次数据处理的重复开销
- 简化代码结构

#### 挑战：
- 需要修改Trainer的加载逻辑
- 需要处理不同模态的特征维度获取（`get_feature()`方法）
- 需要重新设计模型初始化参数传递

#### 统一数据集实现示例：

```python
class UnifiedDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config, "unified")
        # 初始化所有模态的处理器
        self._log_processor = LogProcessor(config)
        self._metric_processor = MetricProcessor(config) 
        self._trace_processor = TraceProcessor(config)
        
    def load(self):
        # 统一加载和处理所有模态数据
        # 确保样本数量和标签一致
        pass
        
    @property
    def log_X(self):
        return self.__log_X__
        
    @property 
    def metric_X(self):
        return self.__metric_X__
        
    @property
    def trace_X(self):
        return self.__trace_X__
        
    def get_metric_feature(self):
        return self.metric_feature_dim
        
    def get_trace_feature(self):
        return self.trace_feature_dim
```

但这需要相应修改Trainer的初始化和数据加载逻辑。

## 结论

当前的分模态设计是合理的，因为：
1. 每种模态有独特的处理流程
2. 便于单独调试和优化
3. 支持单模态实验

如果要实现统一数据集，建议保持当前接口不变，在内部统一处理。
