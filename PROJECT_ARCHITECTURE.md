# Project Architecture

本项目基于 Qwen3-4B-Instruct，使用 QLoRA 对中文金融问答数据进行 SFT 微调，
并完成 base vs SFT 的量化评测。整体分为三条主线：

```text
数据准备 -> 训练微调 -> 评测对比
```

## 完整目录

```text
finance-sft-qwen 项目
├── src/
│   ├── data_loader.py             # 数据层：加载、过滤、清洗、去重、切分
│   │   ├── load_jsonl             # 读 JSONL 原始数据
│   │   ├── filter_zh              # 过滤中文子集
│   │   ├── filter_by_deita_score  # 按 deita_score 质量过滤
│   │   ├── clean_output           # 清洗答案前缀/包裹标记
│   │   ├── conversation_to_alpaca # 转 Alpaca 格式
│   │   ├── deduplicate            # 按 instruction+output 去重
│   │   └── split_train_dev_eval   # seed 固定切分 train/dev/eval
│   ├── evaluator.py               # 指标层：生成答案后算分
│   │   ├── normalize_text         # 文本归一化
│   │   ├── _tokens                # 中英混合 token 化
│   │   ├── _lcs_length            # ROUGE-L 的 LCS
│   │   ├── compute_rouge_l        # ROUGE-L precision/recall/F1
│   │   ├── reference_hit          # 覆盖率命中判断
│   │   ├── corpus_bleu            # corpus-level BLEU
│   │   ├── score_one              # 单条样本打分
│   │   └── aggregate_scores       # 汇总指标
│   └── utils.py                   # 工具层
│       ├── write_jsonl            # 写 JSONL
│       └── write_text             # 写文本报告

├── scripts/
│   ├── prepare_data.py            # 数据准备主流程
│   │   └── main                   # 加载 -> 过滤 -> 去重 -> 切分 -> 写文件 -> 报告
│   ├── train_sft.py               # 训练主流程
│   │   ├── SFTConfig / load_config # 参数层
│   │   ├── load_model_and_tokenizer # 模型层
│   │   │   ├── 4bit NF4 量化
│   │   │   ├── LoRA config
│   │   │   └── get_peft_model
│   │   ├── SFTDataCollator        # batch 层
│   │   │   ├── pad input_ids
│   │   │   ├── pad attention_mask
│   │   │   └── pad labels (-100)
│   │   ├── prepare_dataset        # 数据层
│   │   │   ├── format_chat        # chat template
│   │   │   └── tokenize + labels  # assistant 部分参与 loss
│   │   ├── train                  # 训练层
│   │   │   ├── TrainingArguments
│   │   │   ├── Trainer
│   │   │   ├── resume checkpoint
│   │   │   └── save_model
│   │   └── main                   # 总调度
│   ├── evaluate.py                # 评测主流程
│   │   ├── EvalConfig / load_config  # 参数层
│   │   ├── load_generator         # 模型层
│   │   │   ├── base：不传 adapter
│   │   │   └── SFT：传 adapter_path
│   │   ├── sample_eval_rows       # 数据层
│   │   ├── generate_predictions   # 生成层
│   │   │   ├── prompt 构造
│   │   │   ├── tokenize
│   │   │   ├── model.generate
│   │   │   ├── 截取新 token
│   │   │   └── 保存 JSONL
│   │   ├── build_report           # 报告层
│   │   │   ├── score_one
│   │   │   ├── aggregate_scores
│   │   │   └── 输出 Markdown
│   │   └── main                   # 总调度
│   ├── merge_predictions.py       # 合并四卡预测并生成报告
│   ├── predict.py                 # 交互推理
│   └── app.py                     # FastAPI 部署 + 网页对话

├── configs/
│   ├── train_sft.yaml             # 训练参数
│   └── eval.yaml                  # 评测参数

├── data/
│   ├── raw/                       # 原始 BAAI 数据
│   └── processed/                 # train/dev/eval JSONL

└── outputs/
    ├── checkpoints/final/         # SFT 模型权重
    ├── predictions/               # base/SFT 预测
    └── reports/                   # 数据报告、训练指标、评测报告
```

## 当前状态

- 数据准备：已完成，train/dev/eval = 30000/2000/2000
- SFT 训练：已完成，QLoRA，最终训练 loss 约 1.08
- 指标库：已完成，ROUGE-L / BLEU / Reference Hit
- 评测：已完成，ROUGE-L F1 0.191 -> 0.363，BLEU 0.099 -> 0.295
- 交互推理：已完成
- Web 部署：已完成，FastAPI 网页对话
