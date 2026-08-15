# 学习理解进度

更新时间：2026-08-12

## 已理解

### prepare_data.py

- 整体数据流：
  `原始 jsonl -> 过滤中文 -> 去重 -> 随机切分 -> 转 Alpaca 格式 -> 写文件`
- `ROOT = Path(__file__).resolve().parents[1]`
  - 当前脚本的上级目录，也就是项目根目录
  - 加入 `sys.path` 是为了能导入 `src` 下的模块
- `load_jsonl(RAW_PATH)`
  - 逐行读取 JSONL
  - 每一行都是一个 Python `dict`
- `filter_zh(zh)`
  - 只保留 `lang == "zh"` 的数据
- `deduplicate(zh)`
  - 用 `instruction + "\x00" + output` 做 SHA-256
  - 通过 `seen` 集合去重
- `split_train_dev_eval`
  - 固定 `seed = 42` 打乱
  - 按位置切成 train / dev / eval
  - 好处：可复现，避免原始文件顺序造成数据偏科
- `conversation_to_alpaca`
  - BAAI 的 `conversations` 转成 `instruction / input / output`
  - human 内容作为 instruction
  - gpt 内容作为 output
  - 空 human 或空 gpt 返回 `None`

### data_loader.py

- `load_jsonl`
  - 读取 UTF-8 JSONL
  - 跳过空行
- `filter_zh`
  - 用 `.get("lang")` 防止字段缺失时报错
- `conversation_to_alpaca`
  - 从 `conversations` 中提取 human 和 gpt
  - 额外保留 `id`、`deita_score`、`length`
- `deduplicate`
  - 使用哈希作为集合 key
  - `\x00` 防止不同字符串拼接后碰撞
- `split_train_dev_eval`
  - 固定 seed，确定性打乱
  - 切分后 train/dev/eval 不重叠

## 尚未理解

- `deita_score` 的分布与过滤策略
- `train_sft.py`
- `evaluate.py`
- LLM judge 评测
- 部署 API

## 自我验收

- 能不看代码讲清 `prepare_data.py` 的完整数据流
- 能解释为什么 train/dev/eval 要先打乱再切
- 能解释 deduplicate 为什么用哈希和 `\x00`
