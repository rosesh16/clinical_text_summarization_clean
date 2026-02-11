import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_PATH = Path("data/processed/arxiv/arxiv_val.json")
OUTPUT_PATH = Path("experiments/baseline_results/arxiv_textrank.json")

TOP_K = 3
MAX_SENTENCES = 100   # limit for speed


def textrank(sentences):
    if len(sentences) == 0:
        return ""

    # Cap long documents
    sentences = sentences[:MAX_SENTENCES]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)

    sim_matrix = cosine_similarity(tfidf)
    np.fill_diagonal(sim_matrix, 0)

    scores = sim_matrix.sum(axis=1)
    top_idx = np.argsort(scores)[::-1][:TOP_K]

    return " ".join([sentences[i] for i in sorted(top_idx)])


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    results = []

    for doc in docs:
        sentences = []
        for chunk in doc["chunks"]:
            sentences.extend(chunk["sentences"])

        summary = textrank(sentences)

        results.append({
            "doc_id": doc["doc_id"],
            "summary": summary
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[Fast TextRank] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()