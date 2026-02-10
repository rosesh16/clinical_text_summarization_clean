import json
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    AdamW
)
from tqdm import tqdm


# ---- path injection (consistent with project style) ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


# ---------------- CONFIG ----------------
TRAIN_JSON = "data/processed/pubmed/pubmed_train.json"
OUTPUT_DIR = "checkpoints/bart_pubmed"

MODEL_NAME = "facebook/bart-large-cnn"

MAX_INPUT_TOKENS = 512
MAX_TARGET_TOKENS = 150

BATCH_SIZE = 1                 # RTX 3050 safe
GRAD_ACCUM_STEPS = 4
EPOCHS = 1                     # start with 1; increase to 2 if stable
LR = 2e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------


def build_chunk_text(chunk):
    """Concatenate sentences in a chunk."""
    return " ".join(chunk["sentences"])


def main():
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LR)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        documents = json.load(f)

    step = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for doc in tqdm(documents):
            gold_summary = doc["source"]

            for chunk in doc["chunks"]:
                if len(chunk["sentences"]) == 0:
                    continue

                input_text = build_chunk_text(chunk)

                inputs = tokenizer(
                    input_text,
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                    return_tensors="pt"
                )

                targets = tokenizer(
                    input_text,
                    truncation=True,
                    max_length=MAX_TARGET_TOKENS,
                    return_tensors="pt"
                )

                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                labels = targets["input_ids"].to(DEVICE)

                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=labels
                )

                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()

                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1

        print(f"Completed epoch {epoch + 1}")

    # ---- save fine-tuned model ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nSaved fine-tuned BART model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()