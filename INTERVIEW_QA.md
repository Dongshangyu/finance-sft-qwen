# Interview Q&A

这份文档按面试官可能问的顺序整理，先讲项目背景，再深入技术细节。

## 1. 这个项目做了什么？

我基于 Qwen3-4B-Instruct，使用 QLoRA 在 30,000 条中文金融问答数据上做了 SFT 微调，然后在 2,000 条未见过的评测问题上对比 base 和 SFT 模型，最终量化结果是 ROUGE-L F1 从 0.191 提升到 0.363，BLEU 从 0.099 提升到 0.295。

项目包含数据准备、训练、评测、交互推理四部分，全部代码可复现。

## 2. 为什么选 QLoRA，而不是全量微调？

Qwen3-4B 全量微调需要更新约 40 亿参数，显存和训练成本都很高。QLoRA 把权重冻结，只训练低秩分解矩阵 A 和 B，实际可训练参数只有约 3300 万，占全部参数的 0.8%。

同时模型权重用 4-bit NF4 量化加载，进一步降低显存。这样一张 24GB 的 4090 就能训练 4B 模型。

## 3. LoRA 的 r 和 alpha 是什么？

r 是低秩矩阵的秩，决定可训练参数量和表达能力。alpha 是缩放系数，LoRA 更新量会乘上 alpha / r。我使用 r=16、alpha=32，也就是 2 倍缩放。

r 太小表达能力不足，r 太大训练成本和过拟合风险都会增加。

## 4. 数据是怎么准备的？

原始数据是 BAAI 的金融指令数据，我的流程是：

1. 提取中文子集。
2. 按 instruction + output 去重。
3. 用 deita_score 做质量过滤。
4. 清洗答案里的 `答案：`、`<回答>:` 等格式噪声。
5. 转成 Alpaca 格式：instruction / input / output。
6. 固定 seed 打乱，切分为 train / dev / eval = 30000 / 2000 / 2000。

数据清洗很重要，因为如果答案开头带 `<回答>:`，模型会学到输出这种噪声格式。

## 5. 训练时 labels 是怎么处理的？

我用 chat template 把指令和答案拼成对话，然后把 prompt 部分的 token 对应的 label 设为 -100，只有 assistant 答案部分参与 loss 计算。

-100 是 PyTorch 交叉熵损失约定忽略的位置。这样模型只需要学习怎么回答问题，不需要学习重复问题文本。

## 6. 为什么自己实现 SFTDataCollator？

一个 batch 里的样本长度不同，需要 padding 到同一长度。默认的 collator 会 pad input_ids 和 attention_mask，但 labels 需要特殊处理：padding 位置必须补 -100，不能补 pad token，否则 padding 位置会参与 loss。

自定义 collator 同时 padding 三个字段，保证 batch 维度一致且 loss 计算正确。

## 7. 训练 loss 最终是多少？怎么看？

最终训练 loss 约 1.08。训练 loss 只说明模型在训练集上的预测能力，不能直接说明泛化能力。

所以项目重点放在 2,000 条未见评测数据上的 base vs SFT 对比，而不是只看训练 loss。

## 8. 评测用了什么指标？

我用了 ROUGE-L、BLEU、Reference Hit，全部自己实现。

- ROUGE-L：基于最长公共子序列，计算 precision、recall、F1，衡量生成答案和参考答案的顺序重叠度。
- BLEU：统计 1-4 gram 命中率，并加 brevity penalty 惩罚过短答案。我用的是 corpus-level BLEU。
- Reference Hit：判断生成答案是否覆盖参考答案 30% 以上的核心内容，并排除过短回答。

## 9. 为什么 ROUGE-L 用 LCS 而不是简单字符重合？

LCS 是 Longest Common Subsequence，允许中间插入无关内容，只要核心 token 按顺序出现就算匹配。参考答案是“利率上升，企业融资成本增加”，模型输出“如果利率上升，那么企业融资成本会增加”，LCS 仍能匹配核心内容。

如果要求连续子串，中间插入的连接词会打断匹配，分数会被不合理地压低。

## 10. 为什么评测结果里 base 答案很长？

base 是通用指令模型，生成金融答案时比较啰嗦，平均 886 字符。SFT 训练数据里的答案更精炼，所以微调后平均长度降到 277 字符。

体现在指标上就是 precision 大幅提升：base 0.135，SFT 0.378。SFT 用更短的答案覆盖了和 base 相近比例的核心内容。

## 11. 评测数据为什么按行顺序对齐？

原始数据的 id 字段是模板名，例如 generate_from_given_topic，不是唯一样本标识。2000 条数据只有 3 个不同 id。

如果按 id 对齐，2000 条会被当成 3 条。所以 base 和 SFT 在同一份 eval 数据、同一个 worker 划分下生成，按行顺序对齐是可靠的做法。

## 12. 多卡评测怎么做的？

我按 `index % num_workers == worker_index` 把 2000 条评测样本切成 4 份，每张卡跑 500 条。每个 worker 生成 base 和 SFT 预测，文件分别保存。

全部完成后，merge_predictions.py 把四个 worker 的文件合并成 2000 条，再统一计算指标。

## 13. 推理慢怎么办？

语言模型生成是逐 token 自回归过程，必须串行。我的优化是批量生成：一次喂 8 条 prompt，用 left padding 对齐，比单条生成快很多。

多卡场景再加 worker 切分，四张卡可以并行跑。

## 14. 为什么批量生成要用 left padding？

decoder-only 模型从左向右生成。右 padding 时，padding token 在真实内容之后，会混进生成结果。

left padding 让真实内容靠右，模型生成新 token 时不会把 padding 当成上下文输出。

## 15. 如果换更大的模型，哪些地方要调整？

- 量化方式可能要换成更省显存的配置，或减少 batch size。
- LoRA rank 可能需要调大以增强表达能力。
- 评测生成上限 max_new_tokens 要按任务答案长度调整。
- 多卡推理要考虑显存均衡和通信开销。

## 16. 这个项目体现了哪些工程能力？

- 数据处理：过滤、去重、清洗、固定 seed 切分、质量报告。
- 训练工程：QLoRA、自定义 collator、断点续训、训练日志。
- 评测工程：量化指标实现、base vs SFT 对比、多卡并行、增量保存。
- 可复现性：配置文件和代码分离，结果可复现。

## 17. 项目最大的坑是什么？

一个是评测数据 id 不唯一导致报告只统计 3 条，这个靠检查唯一 id 数量发现。另一个是 decoder-only 批量生成必须用 left padding，否则生成结果会被污染。

这两个问题都说明：跑通不等于正确，评测必须验证对齐和生成细节。
