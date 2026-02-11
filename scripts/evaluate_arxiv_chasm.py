import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from rouge_score import rouge_scorer
from bert_score import score as bertscore


# Paths
GOLD_PATH = Path("data/raw/arxiv/arxiv.json")
PRED_PATH = Path("experiments/baseline_results/chasm_full_arxiv.json")
OUTPUT_PATH = Path("experiments/metrics/arxiv_chasm_metrics.json")


def compute_redundancy(text):
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return 0.0

    unique_sentences = set(sentences)
    redundancy = 1 - (len(unique_sentences) / len(sentences))
    return redundancy


def main():
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold_data = {d["doc_id"]: d["summary"] for d in json.load(f)}

    with open(PRED_PATH, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    rouge_scores = []
    redundancies = []
    gold_texts = []
    pred_texts = []

    for item in tqdm(pred_data):
        doc_id = item["doc_id"]
        pred_summary = item["summary"]
        gold_summary = gold_data.get(doc_id, "")

        if not gold_summary.strip():
            continue

        # ROUGE-1
        rouge = scorer.score(gold_summary, pred_summary)
        rouge_scores.append(rouge["rouge1"].fmeasure)

        # Redundancy
        redundancies.append(compute_redundancy(pred_summary))

        # BERTScore input
        gold_texts.append(gold_summary)
        pred_texts.append(pred_summary)

    # BERTScore (GPU will be used automatically)
    P, R, F1 = bertscore(pred_texts, gold_texts, lang="en", verbose=True)

    metrics = {
        "ROUGE-1": float(np.mean(rouge_scores)),
        "BERTScore_F1": float(F1.mean()),
        "Redundancy": float(np.mean(redundancies))
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== CHASM arXiv Evaluation ===")
    print(metrics)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()