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

- 首选：`Qwen/Qwen3-4B-Instruct`
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

## 10 天执行计划

### Day 1-2：数据准备与工程骨架

- 建 PyCharm 工程
- 下载并检查 BAAI 中文金融数据
- 完成去重、过滤、train/dev/eval 划分
- 输出 `data_report.md`

验收：

- 数据量、来源、字段、样本示例记录清楚
- train/dev/eval 无重叠

### Day 3：环境与 smoke test

- 在 AutoDL 安装依赖
- 用 200 条数据跑通 QLoRA-SFT
- 确认 loss 下降、adapter 可保存、可加载

验收：

- smoke 训练能完成
- 能看到训练日志和 checkpoint

### Day 4-6：正式训练与评测

- 用约 30,000 条数据训练 1 epoch
- 保存最终 LoRA adapter
- 在同一评测集上生成 base 和 SFT 的回答
- 完成 LLM judge 和 ROUGE-L 评测

验收：

- 训练日志完整
- 评测分数表完成
- 10 组 before/after 回答对比完成

### Day 7-8：优化与失败分析

- 根据失败案例调整数据过滤、学习率、LoRA rank 或 epoch
- 可选跑 1 个对比实验
- 写出“为什么变好/为什么没变好”

验收：

- 至少有一版明显优于 base 的模型
- 失败案例分析完成

### Day 9：部署

- 写 FastAPI 推理接口
- 加载 LoRA adapter
- 完成至少 5 个端到端测试

验收：

- 可通过 API 问答
- 记录 GPU 占用和响应时间

### Day 10：项目收尾

- 写 README
- 写项目技术报告
- 准备面试问答
- 生成简历项目描述

验收：

- 不查资料能讲清完整链路
- 简历上的每个数字都有对应实验证据

## 风险与取舍

- 如果 Qwen3-4B 训练出现兼容问题，切换 Qwen2.5-3B
- 如果数据量太大导致训练时间不够，优先减少 epoch，不降低评测质量
- 如果 LLM judge 不稳定，用人工抽评 + ROUGE-L 双轨
- 本项目不做 DPO、RAG、8B，先把 SFT 和量化评测做扎实
