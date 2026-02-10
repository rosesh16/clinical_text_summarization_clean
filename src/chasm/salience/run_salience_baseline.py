import json
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.chasm.salience.salience_model import SalienceFusion
from src.chasm.salience.scorers import normalize, position_prior


VAL_JSON = "data/processed/pubmed/pubmed_val.json"
TEXTRANK_JSON = "experiments/baseline_results/textrank.json"
BERTSUM_JSON = "experiments/baseline_results/bertsum.json"
OUTPUT_JSON = "experiments/baseline_results/chasm_salience.json"

TOP_K = 3   # match Lead-3 for fairness


def load_baseline_scores(path):
    """
    Returns dict: doc_id -> list of sentences (in order)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {d["doc_id"]: d["summary"] for d in data}


def main():
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)

    textrank = load_baseline_scores(TEXTRANK_JSON)
    bertsum = load_baseline_scores(BERTSUM_JSON)

    model = SalienceFusion()
    model.eval()

    results = []

    for doc in docs:
        doc_id = doc["doc_id"]

        # ---- collect sentences in original order ----
        sentences = []
        for c in doc["chunks"]:
            sentences.extend(c["sentences"])

        if not sentences:
            summary = ""
            results.append({"doc_id": doc_id, "summary": summary})
            continue

        N = len(sentences)

        # ---- proxy scores ----
        # TextRank / BERTSUM summaries are sentence subsets,
        # we convert them into weak per-sentence scores
        textrank_scores = np.array([
            1.0 if s in textrank.get(doc_id, "") else 0.0
            for s in sentences
        ])

        bertsum_scores = np.array([
            1.0 if s in bertsum.get(doc_id, "") else 0.0
            for s in sentences
        ])

        pos_scores = position_prior(N)

        scores = model(
            torch_tensor(normalize(textrank_scores)),
            torch_tensor(normalize(bertsum_scores)),
            torch_tensor(normalize(pos_scores))
        ).detach().numpy()

        topk_idx = np.argsort(scores)[::-1][:TOP_K]
        summary = " ".join([sentences[i] for i in sorted(topk_idx)])

        results.append({
            "doc_id": doc_id,
            "summary": summary
        })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved CHASM-Salience baseline to {OUTPUT_JSON}")


def torch_tensor(x):
    import torch
    return torch.tensor(x, dtype=torch.float)


if __name__ == "__main__":
    main()