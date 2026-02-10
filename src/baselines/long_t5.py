import json
import os
import sys
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

MODEL_NAME_OR_PATH = "google/long-t5-tglobal-base"
INPUT_JSON = "data/processed/pubmed/pubmed_val.json"
OUTPUT_JSON = "experiments/baseline_results/long_t5.json"

MAX_INPUT_TOKENS = 2048
MAX_OUTPUT_TOKENS = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def summarize_text(text, tokenizer, model):
    inputs = tokenizer(
        "summarize: " + text,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        return_tensors="pt"
    ).to(DEVICE)

    summary_ids = model.generate(
    inputs["input_ids"],
    max_length=MAX_OUTPUT_TOKENS,
    min_length=40,
    num_beams=4,
    length_penalty=1.5,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    early_stopping=True,
)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def main():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME_OR_PATH)
    model.to(DEVICE)
    model.eval()

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        documents = json.load(f)

    results = []

    with torch.no_grad():
        for doc in tqdm(documents):
            chunk_summaries = []

            chunks = doc["chunks"]
            for i in range(0, len(chunks) - 2):
                merged_text = " ".join(
                    " ".join(c["sentences"]) for c in chunks[i:i+3]
                    if len(c["sentences"]) > 0
                )

                if not merged_text.strip():
                    continue

                summary = summarize_text(merged_text, tokenizer, model)
                if summary.strip() and len(summary.split()) > 5:
                  chunk_summaries.append(summary)

            final_summary = " ".join(chunk_summaries)

            results.append({
                "doc_id": doc["doc_id"],
                "summary": final_summary
            })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved Long-T5 baseline to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()