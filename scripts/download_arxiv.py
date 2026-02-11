# scripts/download_arxiv_clean.py

import os
import json
from datasets import load_dataset

OUTPUT_PATH = "data/raw/arxiv/arxiv.json"
NUM_SAMPLES = 500

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    dataset = load_dataset(
        "ccdv/arxiv-summarization",
        split="validation"
    )

    data = []

    for i, sample in enumerate(dataset):
        if i >= NUM_SAMPLES:
            break

        data.append({
            "doc_id": f"ARXIV_{i}",
            "source": "arxiv",
            "title": sample.get("title", ""),
            "text": sample["article"],
            "summary": sample["abstract"],
            "meta": {
                "abstract_only": False
            }
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} samples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()