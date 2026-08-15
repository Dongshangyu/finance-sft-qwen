# 项目问题合集

用途：记录做项目时遇到的问题、原因和处理方法，方便最后复习项目和给面试官讲。
规则：遇到新问题就往这里追加，保留“问题 -> 原因 -> 处理 -> 面试怎么讲”的结构。

## 1. BAAI 原始数据格式和训练格式不一致

问题：

- BAAI 原始数据是 `conversations` 格式：
```json
{
  "conversations": [
    {"from": "human", "value": "问题"},
    {"from": "gpt", "value": "答案"}
  ]
}
```

- SFT 训练更常用 Alpaca 格式：
```json
{
  "instruction": "问题",
  "input": "",
  "output": "答案"
}
```

原因：

- 不同数据集和训练框架的数据格式并不统一。

处理：

- 在 `data_loader.py` 中实现 `conversation_to_alpaca`。
- 从 `conversations` 中提取 `human` 作为 `instruction`，`gpt` 作为 `output`。
- 数据不足时返回 `None`，直接丢弃不合法样本。

面试怎么讲：

- 能说清原始数据字段如何映射到训练字段。
- 能说明空字段为什么需要过滤。
- 数据格式转换是数据工程里最基础也最重要的一步。

## 2. 原始答案存在大量格式噪声

问题：

- 抽样时发现答案开头有：
  - `<回答>:`
  - `%ANSWER%:`
  - `答案：...`
  - 整个答案被 `<...>` 包住

原因：

- BAAI 是自动化生成/清洗的金融指令数据，部分答案保留了生成模板标记。

处理：

- 在 `data_loader.py` 中新增 `clean_output`。
- 去掉开头的 `答案：`、`回答：`、`<回答>:`、`%ANSWER%:`、`%QUERY%:`、`Answer:` 等标记。
- 如果整个答案被 `<...>` 包住，只去掉最外层包裹。
- 转换 Alpaca 格式时自动调用，重新生成 train/dev/eval。

清理前统计：

- 约 26% 答案从 `答案：` 开头
- 约 14.3% 答案被整段 `<...>` 包住

清理后统计：

- train/dev/eval 中残留前缀 0
- 整段包裹 0

面试怎么讲：

- 数据清洗会直接影响模型输出格式。
- 如果不清洗，模型可能学会输出 `<回答>:` 这类噪声。
- 用数量统计证明清洗效果，而不是只说“我清洗了一下”。

## 3. 如何保证数据划分可复现

问题：

- 随机切分如果每次结果不同，训练和评测结果就不好复现。

处理：

- 使用固定 `seed = 42` 打乱，再切 train/dev/eval。

面试怎么讲：

- 固定 seed 保证每次运行得到相同的数据划分。
- 先打乱再切，避免原始文件顺序造成数据偏移。
- 例如原始文件如果前面全是银行类、后面全是投资类，直接顺序切分会让评测集偏移。

## 4. 模型 ID 写错导致 Hugging Face 下载失败

问题：

- `Qwen/Qwen3-4B-Instruct` 这个仓库 ID 在 Hugging Face 上不存在。
- 官方模型 ID 是 `Qwen/Qwen3-4B-Instruct-2507`。

原因：

- `from_pretrained` 里的字符串会被当成本地路径或 Hugging Face 仓库 ID，仓库不存在就报 OSError。

处理：

- 把 `configs/train_sft.yaml`、`configs/eval.yaml`、脚本默认值统一改为 `Qwen/Qwen3-4B-Instruct-2507`。

面试怎么讲：

- 能说出 `from_pretrained` 的输入可以是本地路径或 Hugging Face 仓库 ID。
- 模型版本后缀要核对官方仓库，不能凭记忆写。

## 5. Hugging Face xet 下载报 401

问题：

- 新版 Hugging Face 默认使用 xet 加速下载，在 hf-mirror 上可能失败，报 CAS Client Error 401。

原因：

- xet 传输服务无法访问或镜像源不支持。

处理：

- 下载或运行前设置 `export HF_HUB_DISABLE_XET=1`，强制走普通 HTTP 下载。
- 国内网络再配合 `export HF_ENDPOINT=https://hf-mirror.com`。

面试怎么讲：

- 能解释大模型权重通过 cache 下载，镜像源和传输协议可能造成失败。
- 能用环境变量控制下载行为和下载源。

## 6. decoder-only 模型批量生成必须用 left padding

问题：

- 批量生成多条 prompt 时，tokenizer 默认右 padding，报 right-padding warning。

原因：

- decoder-only 模型从左向右生成，padding 放右边会混进生成结果，污染预测。

处理：

- 加载 tokenizer 后设置 `tokenizer.padding_side = "left"`。

面试怎么讲：

- 能说明 padding 位置对因果语言模型生成的影响。
- 能解释 batch 推理时如何正确对齐不同长度的 prompt。

## 7. 评测数据 id 不唯一导致报告只统计 3 条

问题：

- eval.jsonl 有 2000 条，但只有 3 个不同的 id，报告 eval samples 显示 3。

原因：

- 原始数据里的 id 是模板名，不是唯一样本标识。
- 早期 build_report 按 id 字典对齐，重复 id 被覆盖。

处理：

- 改为按 base 和 SFT 预测文件的行顺序对齐。
- 后续数据准备阶段应生成唯一样本 id。

面试怎么讲：

- 能说明评测对比前必须确认两条预测对应同一个问题。
- 能识别数据字段命名陷阱：id 不一定是唯一主键。

## 8. 单条生成太慢，改为批量生成 + 多卡 worker

问题：

- 2000 条评测单条生成非常慢，几小时只跑几百条。

原因：

- 语言模型生成是逐 token 串行过程。
- 旧脚本一次只生成一条 prompt，GPU 利用率低。

处理：

- 评测脚本增加 batch_size，一次生成多条。
- 增加 --worker-index / --num-workers，四张卡各跑 500 条。
- 全部完成后用 merge_predictions.py 合并并生成报告。

面试怎么讲：

- 能说明推理和训练的计算模式不同，推理是自回归串行生成。
- 能说明批量推理、多卡并行、断点续跑如何提升吞吐。

## 9. 终端输入带无效 surrogate 导致 tokenizer 崩溃

问题：

- 交互预测时，终端里删除重打输入会偶发 `\udce5` 这类无效 Unicode。
- `apply_chat_template` 原样保留，tokenizer 编码时报 `TextEncodeInput must be Union[...]`。

原因：

- SSH 终端输入在编辑时可能产生 invalid surrogate。
- transformers 4.52.4 的 tokenizer 不支持 `errors="ignore"` 参数。

处理：

- 在 `input()` 后过滤 surrogate 字符，再进入 chat template。
- 不要在 tokenizer 调用里传 `errors="ignore"`，旧版本不支持。

面试怎么讲：

- 能说明真实项目会遇到终端输入编码污染，不是模型问题。
- 能说明输入清洗应该放在进入 tokenizer 之前。
