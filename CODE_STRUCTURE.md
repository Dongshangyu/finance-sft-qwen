# 代码结构

```text
finance-sft-qwen/
├── README.md
├── PLAN.md
├── CODE_STRUCTURE.md
├── requirements.txt
├── configs/
│   ├── train_sft.yaml
│   └── eval.yaml
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── prepare_data.py
│   ├── train_sft.py
│   ├── evaluate.py
│   ├── predict.py
│   └── app.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── judge.py
│   └── utils.py
└── outputs/
    ├── checkpoints/
    ├── predictions/
    ├── logs/
    └── reports/
```

## 模块职责

### scripts/prepare_data.py

负责：

- 读取 BAAI 原始 jsonl
- 过滤中文
- 去重
- 按 deita_score 过滤
- 划分 train / dev / eval
- 保存到 `data/processed/`

入口：

```python
def prepare_data(raw_dir: str, output_dir: str) -> None:
    ...
```

### scripts/train_sft.py

负责：

- 读取训练配置
- 加载 Qwen3-4B-Instruct
- 配置 QLoRA
- 用 Hugging Face Trainer 训练
- 保存 adapter 和训练日志

入口：

```python
def train(config_path: str) -> None:
    ...
```

### scripts/evaluate.py

负责：

- 加载评测集
- 分别用 base 和 SFT 生成回答
- 计算 LLM judge 分数和 ROUGE-L
- 输出分数表和 before/after 案例

入口：

```python
def evaluate(config_path: str) -> None:
    ...
```

### scripts/predict.py

负责：

- 加载最终 adapter
- 输入 prompt，输出回答
- 用于本地交互测试

### scripts/app.py

负责：

- FastAPI 服务
- 接收问题，返回回答
- 记录响应时间

### src/data_loader.py

封装数据读取、过滤、划分逻辑。

### src/evaluator.py

封装评测指标：LLM judge、ROUGE-L、回答长度等。

### src/judge.py

封装 LLM judge 的 prompt 和打分解析。

### src/utils.py

统一日志、文件读写、JSONL 读写等工具。

## 关键约定

- 原始数据只放在 `data/raw/`
- 处理后的数据只放在 `data/processed/`
- 模型输出只放在 `outputs/predictions/`
- 评测报告只放在 `outputs/reports/`
- 所有训练和评测结果都要记录对应配置
