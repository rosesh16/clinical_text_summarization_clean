import json
import os
import sys
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration
from tqdm import tqdm

# ---- path injection ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

MODEL_NAME = "allenai/led-base-16384"
INPUT_JSON = "data/processed/pubmed/pubmed_val.json"
OUTPUT_JSON = "experiments/baseline_results/led.json"

MAX_INPUT_TOKENS = 4096    # safe on RTX 3050
MAX_OUTPUT_TOKENS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    tokenizer = LEDTokenizer.from_pretrained(MODEL_NAME)
    model = LEDForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        documents = json.load(f)

    results = []

    with torch.no_grad():
        for doc in tqdm(documents):
            # ---- build full document text ----
            sentences = []
            for chunk in doc["chunks"]:
                sentences.extend(chunk["sentences"])

            if not sentences:
                summary = ""
            else:
                text = " ".join(sentences)

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS
                )

                input_ids = inputs["input_ids"].to(DEVICE)
                attention_mask = inputs["attention_mask"].to(DEVICE)

                # ---- global attention on first token ----
                global_attention_mask = torch.zeros_like(attention_mask)
                global_attention_mask[:, 0] = 1

                summary_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    max_length=MAX_OUTPUT_TOKENS,
                    num_beams=4,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

                summary = tokenizer.decode(
                    summary_ids[0],
                    skip_special_tokens=True
                )

            results.append({
                "doc_id": doc["doc_id"],
                "summary": summary
            })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved LED baseline to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()