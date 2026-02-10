import json
import os

INPUT_JSON = "experiments/baseline_results/chasm_factchecked.json"
OUTPUT_JSON = "experiments/baseline_results/chasm_factaware.json"

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    stripped = [
        {
            "doc_id": item["doc_id"],
            "summary": item["summary"]
        }
        for item in data
    ]

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stripped, f, indent=2)

    print(f"Saved summary-only fact-aware CHASM to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()