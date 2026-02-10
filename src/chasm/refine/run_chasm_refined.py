import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.chasm.refine.rewriter import ClaimRewriter


FACT_JSON = "experiments/baseline_results/chasm_factchecked.json"
OUTPUT_JSON = "experiments/baseline_results/chasm_refined.json"

TOP_K_EVIDENCE = 5


def main():
    with open(FACT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    rewriter = ClaimRewriter()
    outputs = []

    for item in data:
        refined_claims = []

        for c in item["claims"]:
            if c["supported"]:
                refined_claims.append(c["text"])
            else:
                rewritten = rewriter.rewrite(
                    c["text"],
                    [ev["text"] for ev in item["claims"]][:TOP_K_EVIDENCE]
                )
                refined_claims.append(rewritten)

        outputs.append({
            "doc_id": item["doc_id"],
            "summary": " ".join(refined_claims)
        })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print(f"Saved self-refined CHASM output to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()