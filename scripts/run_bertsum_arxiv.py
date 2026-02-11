import json
from pathlib import Path
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.chasm.generator.bart_generator import BartGenerator

INPUT_PATH = Path("data/processed/arxiv/arxiv_val.json")
OUTPUT_PATH = Path("experiments/baseline_results/arxiv_bertsum.json")

TOP_K = 5

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    generator = BartGenerator()

    results = []

    for doc in docs:
        sentences = []
        for chunk in doc["chunks"]:
            sentences.extend(chunk["sentences"])

        if not sentences:
            summary = ""
        else:
            input_text = " ".join(sentences[:TOP_K])
            summary = generator.generate(
                input_text,
                max_len=150,
                min_len=40
            )

        results.append({
            "doc_id": doc["doc_id"],
            "summary": summary
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[BERTSUM] Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()