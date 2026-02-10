import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.chasm.fact_check.verifier import FactVerifier


CHASM_INPUT = "experiments/baseline_results/chasm_full.json"
VAL_JSON = "data/processed/pubmed/pubmed_val.json"
OUTPUT_JSON = "experiments/baseline_results/chasm_factchecked.json"

TOP_M_EVIDENCE = 10   # salient sentences used as evidence


def main():
    with open(CHASM_INPUT, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    with open(VAL_JSON, "r", encoding="utf-8") as f:
        docs = {d["doc_id"]: d for d in json.load(f)}

    verifier = FactVerifier()

    outputs = []

    for item in summaries:
        doc_id = item["doc_id"]
        summary = item["summary"]

        # collect evidence sentences
        evidence = []
        for c in docs[doc_id]["chunks"]:
            evidence.extend(c["sentences"])
        evidence = evidence[:TOP_M_EVIDENCE]

        claim_results, conf = verifier.verify_summary(summary, evidence)

        outputs.append({
            "doc_id": doc_id,
            "summary": summary,
            "fact_confidence": round(conf, 4),
            "claims": [
                {
                    "text": c,
                    "entailment": round(score, 4),
                    "supported": ok
                }
                for c, score, ok in claim_results
            ]
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print(f"Saved fact-verified CHASM output to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()