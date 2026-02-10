import json
import os
import sys
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

MODEL_NAME_OR_PATH = "google/pegasus-pubmed"
INPUT_JSON = "data/processed/pubmed/pubmed_val.json"
OUTPUT_JSON = "experiments/baseline_results/pegasus.json"

MAX_INPUT_TOKENS = 512
MAX_OUTPUT_TOKENS = 120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def summarize_chunk(text, tokenizer, model):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        return_tensors="pt"
    ).to(DEVICE)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=MAX_OUTPUT_TOKENS,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def main():
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME_OR_PATH)
    model.to(DEVICE)
    model.eval()

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        documents = json.load(f)

    results = []

    with torch.no_grad():
        for doc in tqdm(documents):
            chunk_summaries = []

            for chunk in doc["chunks"]:
                if len(chunk["sentences"]) == 0:
                    continue

                chunk_text = " ".join(chunk["sentences"])
                summary = summarize_chunk(chunk_text, tokenizer, model)
                chunk_summaries.append(summary)

            final_summary = " ".join(chunk_summaries)

            results.append({
                "doc_id": doc["doc_id"],
                "summary": final_summary
            })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved PEGASUS baseline to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()