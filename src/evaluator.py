"""Evaluation metrics for base vs SFT comparison."""

from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean


def normalize_text(text: str) -> str:
    """Normalize Chinese and English text before scoring."""
    text = text.replace("\u3000", " ").replace("，", ",")
    text = text.replace("。", ".").replace("；", ";")
    text = text.replace("：", ":").replace("、", ",")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _tokens(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text into stable score tokens."""
    text = normalize_text(text)
    english = re.findall(r"[a-z0-9_]+", text)
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return english + chinese


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length used by ROUGE-L."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-L precision, recall, and F1 over score tokens."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens or not ref_tokens:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "rouge_l_precision": precision,
        "rouge_l_recall": recall,
        "rouge_l_f1": f1,
    }


def reference_hit(
    prediction: str,
    reference: str,
    min_recall: float = 0.3,
    min_answer_len: int = 20,
) -> float:
    """Return 1 when the prediction covers at least 30% of the reference."""
    if len(prediction.strip()) < min_answer_len:
        return 0.0
    rouge = compute_rouge_l(prediction, reference)
    return 1.0 if rouge["rouge_l_recall"] >= min_recall else 0.0


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    predictions: list[str],
    references: list[str],
    max_order: int = 4,
) -> float:
    """Compute corpus-level BLEU with a single brevity penalty."""
    if not predictions:
        return 0.0

    matches_by_order = [0] * max_order
    possible_by_order = [0] * max_order
    total_ref_len = 0
    total_pred_len = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = _tokens(prediction)
        ref_tokens = _tokens(reference)
        total_pred_len += len(pred_tokens)
        total_ref_len += len(ref_tokens)
        if not pred_tokens or not ref_tokens:
            continue

        for n in range(1, max_order + 1):
            pred_counts = _ngrams(pred_tokens, n)
            ref_ngrams = _ngrams(ref_tokens, n)
            for ngram, count in pred_counts.items():
                possible_by_order[n - 1] += count
                matches_by_order[n - 1] += min(count, ref_ngrams.get(ngram, 0))

    precisions = []
    for n in range(max_order):
        if possible_by_order[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches_by_order[n] / possible_by_order[n])

    if min(precisions) == 0:
        geometric_mean = 0.0
    else:
        geometric_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)

    if total_pred_len == 0 or total_ref_len == 0:
        brevity_penalty = 0.0
    elif total_pred_len > total_ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - total_ref_len / total_pred_len)

    return geometric_mean * brevity_penalty


def score_one(prediction: str, reference: str) -> dict[str, float]:
    """Return per-example scores for a single prediction/reference pair."""
    rouge = compute_rouge_l(prediction, reference)
    return {
        **rouge,
        "reference_hit": reference_hit(prediction, reference),
        "short_prediction": int(len(prediction.strip()) < 20),
        "prediction_len": len(prediction.strip()),
        "reference_len": len(reference.strip()),
    }


def aggregate_scores(rows: list[dict]) -> dict[str, float]:
    """Aggregate per-example score rows into summary metrics."""
    if not rows:
        return {}

    return {
        "count": float(len(rows)),
        "rouge_l_f1": mean(row["rouge_l_f1"] for row in rows),
        "rouge_l_precision": mean(row["rouge_l_precision"] for row in rows),
        "rouge_l_recall": mean(row["rouge_l_recall"] for row in rows),
        "reference_hit": mean(row["reference_hit"] for row in rows),
        "bleu": corpus_bleu(
            [row["prediction"] for row in rows],
            [row["reference"] for row in rows],
        ),
        "mean_prediction_len": mean(row["prediction_len"] for row in rows),
        "mean_reference_len": mean(row["reference_len"] for row in rows),
        "empty_predictions": sum(1 for row in rows if not row["prediction"].strip()),
        "short_predictions": sum(row["short_prediction"] for row in rows),
    }
