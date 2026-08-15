# Base vs SFT Evaluation Report

- eval samples: 2000
- base predictions: `/root/autodl-tmp/finance-sft-qwen/outputs/predictions/base_predictions.jsonl`
- SFT predictions: `/root/autodl-tmp/finance-sft-qwen/outputs/predictions/sft_predictions.jsonl`

| metric | base | SFT | delta |
| --- | ---: | ---: | ---: |
| ROUGE-L F1 | 0.1912 | 0.3627 | +0.1715 |
| ROUGE-L Precision | 0.1349 | 0.3775 | +0.2426 |
| ROUGE-L Recall | 0.3782 | 0.3756 | -0.0026 |
| BLEU | 0.0992 | 0.2948 | +0.1956 |
| Reference Hit | 0.7625 | 0.7465 | -0.0160 |
| Mean Prediction Length | 886.2725 | 276.7985 | -609.4740 |
| Empty Predictions | 0.0000 | 0.0000 | +0.0000 |
| Short Predictions | 0.0000 | 0.0000 | +0.0000 |
