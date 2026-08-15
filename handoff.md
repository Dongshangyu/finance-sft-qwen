# 学习任务存档（Handoff）

> 当前日期：2026-08-12
> 任务目录：`C:\Users\19935\Documents\Codex\learning-tasks\finance-sft-qwen`
> 项目：基于 Qwen3-4B-Instruct 的中文金融问答 SFT 微调与量化评测

## 当前状态

- 已选定数据集：BAAI/IndustryInstruction_Finance-Economics
- 已选定模型方向：Qwen3-4B-Instruct，备选 Qwen2.5-3B-Instruct
- 已生成 `PLAN.md` 和 `CODE_STRUCTURE.md`
- 已搭出 PyCharm 工程骨架
- 原始数据已放入 `data/raw/`

## 下一步

1. 在 PyCharm 打开 `finance-sft-qwen`
2. 实现 `prepare_data.py` 和数据划分
3. 做 200 条 smoke test
4. 跑正式 SFT
5. 完成 base vs SFT 量化评测

## 教学方式

- 中文讲解，技术术语保留英文
- 用户在 PyCharm 中写代码，不通过终端复制粘贴
- 每个模块先讲清原理，再实现和验证
- 最终验收标准：能独立讲清数据、训练、评测、部署全链路
