import json
import os
import sys
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

TRAIN_JSON = "data/processed/pubmed/pubmed_train.json"
OUTPUT_DIR = "checkpoints/long_t5_pubmed"

MODEL_NAME = "google/long-t5-tglobal-base"

MAX_INPUT_TOKENS = 2048
MAX_TARGET_TOKENS = 150
EPOCHS = 1
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LR)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        documents = json.load(f)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")

        for doc in tqdm(documents):
            chunks = doc["chunks"]

            for i in range(0, len(chunks) - 2):
                merged_text = " ".join(
                    " ".join(c["sentences"]) for c in chunks[i:i+3]
                    if len(c["sentences"]) > 0
                )

                if not merged_text.strip():
                    continue

                inputs = tokenizer(
                    "summarize: " + merged_text,
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                    return_tensors="pt"
                )

                targets = tokenizer(
                    merged_text,
                    truncation=True,
                    max_length=MAX_TARGET_TOKENS,
                    return_tensors="pt"
                )

                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                labels = targets["input_ids"].to(DEVICE)

                loss = model(
                    input_ids=inputs["input_ids"],
                    labels=labels
                ).loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Saved fine-tuned Long-T5 to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()