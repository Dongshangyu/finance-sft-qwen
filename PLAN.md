# 金融问答 SFT 微调与量化评测计划

## 项目目标

基于 Qwen3-4B-Instruct，使用中文金融指令数据做 LoRA/QLoRA SFT，最终产出：

- 可复现的数据准备脚本
- 可复现的 SFT 训练配置
- base vs SFT 的量化评测结果
- 可用于部署的推理 API
- 一份能写进简历并扛住面试追问的项目报告

## 数据方案

数据集：`BAAI/IndustryInstruction_Finance-Economics`

- 总条数：122,090
- 中文子集：40,135
- 计划使用：约 30,000 条训练，2,000 条 dev，2,000 条 eval
- 官方 500 条 eval 可作为补充检查

数据准备要求：

- 只保留 `lang == "zh"`
- 保留 `human -> gpt` 问答结构
- 按问题和答案做去重
- 可按 `deita_score` 过滤低质量数据
- 训练集、dev、eval 严格不重叠

## 模型方案

- 首选：`Qwen/Qwen3-4B-Instruct-2507`
- 备选：`Qwen/Qwen2.5-3B-Instruct`
- 训练方式：QLoRA
- 显存：AutoDL RTX 4090 24GB 可跑

## 评测方案

评测集：2,000 条中文金融问答，训练时完全不使用。

指标：

- LLM judge 打分：相关性、完整性、忠实度，每项 1-5
- ROUGE-L 作为参考型指标
- 对适合计算的题目，可额外统计数值正确率
- 每组生成保留原始输出，便于人工检查

最终报告必须包含：

- base vs SFT 的分数表
- 至少 10 组真实回答对比
- 至少 5 个失败案例和原因分析

## 风险与取舍

- 如果 Qwen3-4B 训练出现兼容问题，切换 Qwen2.5-3B
- 如果数据量太大导致训练时间不够，优先减少 epoch，不降低评测质量
- 如果 LLM judge 不稳定，用人工抽评 + ROUGE-L 双轨
- 本项目不做 DPO、RAG、8B，先把 SFT 和量化评测做扎实
