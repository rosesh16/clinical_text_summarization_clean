import json
import os
import sys
import numpy as np

from src.chasm.hierarchy.mmr_selector import MultiDocMMRSelector

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.chasm.salience.salience_model import SalienceFusion
from src.chasm.salience.scorers import normalize, position_prior
from src.chasm.hierarchy.hierarchical_ranker import HierarchicalRanker
from src.chasm.generator.bart_generator import BartGenerator


VAL_JSON = "data/processed/pubmed/pubmed_val.json"
TEXTRANK_JSON = "experiments/baseline_results/textrank.json"
BERTSUM_JSON = "experiments/baseline_results/bertsum.json"
OUTPUT_JSON = "experiments/baseline_results/chasm_full.json"

TOP_K = 8   # more context before abstraction


def load_baseline(path):
    with open(path, "r", encoding="utf-8") as f:
        return {d["doc_id"]: d["summary"] for d in json.load(f)}


def to_tensor(x):
    import torch
    return torch.tensor(x, dtype=torch.float)


def main():
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)

    textrank = load_baseline(TEXTRANK_JSON)
    bertsum = load_baseline(BERTSUM_JSON)

    fusion = SalienceFusion().eval()
    ranker = HierarchicalRanker(max_per_chunk=2)
    generator = BartGenerator()

    results = []

    for doc in docs:
        doc_id = doc["doc_id"]

        sentences = []
        chunk_ids = []
        for cid, c in enumerate(doc["chunks"]):
            for s in c["sentences"]:
                sentences.append(s)
                chunk_ids.append(cid)

        if not sentences:
            results.append({"doc_id": doc_id, "summary": ""})
            continue

        N = len(sentences)

        textrank_scores = np.array([
            1.0 if s in textrank.get(doc_id, "") else 0.0
            for s in sentences
        ])
        bertsum_scores = np.array([
            1.0 if s in bertsum.get(doc_id, "") else 0.0
            for s in sentences
        ])
        pos_scores = position_prior(N)

        scores = fusion(
            to_tensor(normalize(textrank_scores)),
            to_tensor(normalize(bertsum_scores)),
            to_tensor(normalize(pos_scores))
        ).detach().numpy()

        # Track document ids
        doc_ids = []
        for doc_index, doc in enumerate([doc]):  # replace with real multi-doc list later
            for c in doc["chunks"]:
                for _ in c["sentences"]:
                    doc_ids.append(doc_index)

        mmr_selector = MultiDocMMRSelector()

        salient_sents, _ = mmr_selector.select(
            sentences=sentences,
            salience_scores=scores,
            doc_ids=doc_ids,
            top_k=TOP_K,
            lambda_param=0.7,
            max_per_doc=3
        )

        generator_input = " ".join(salient_sents)

        summary = generator.generate(generator_input)

        results.append({
            "doc_id": doc_id,
            "summary": summary
        })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved CHASM full model output to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()