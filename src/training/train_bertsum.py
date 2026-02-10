import json
import os
import sys
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW
from rouge_score import rouge_scorer
from tqdm import tqdm

# ---- path injection (same style as evaluation) ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.baselines.bertsum import BertSumExt


# ---------------- CONFIG ----------------
TRAIN_JSON = "data/processed/pubmed/pubmed_train.json"
VAL_JSON   = "data/processed/pubmed/pubmed_val.json"
OUTPUT_JSON = "experiments/baseline_results/bertsum.json"

MAX_TOKENS = 512
MAX_ORACLE_SENTENCES = 3
LR = 2e-5
EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------


def build_bertsum_input(sentences, tokenizer):
    tokens = []
    cls_positions = []

    for sent in sentences:
        cls_positions.append(len(tokens))
        tokens += [tokenizer.cls_token] + tokenizer.tokenize(sent) + [tokenizer.sep_token]

    tokens = tokens[:MAX_TOKENS]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    return (
        torch.tensor(input_ids).unsqueeze(0),
        torch.tensor(attention_mask).unsqueeze(0),
        torch.tensor(cls_positions).unsqueeze(0),
    )


def build_rouge_oracle(sentences, gold_summary, scorer, max_sentences):
    selected = []
    selected_idx = set()
    current_rouge = 0.0

    for _ in range(min(max_sentences, len(sentences))):
        best_gain = 0.0
        best_i = None

        for i, s in enumerate(sentences):
            if i in selected_idx:
                continue

            rouge = scorer.score(
                " ".join(selected + [s]),
                gold_summary
            )["rougeL"].fmeasure

            gain = rouge - current_rouge
            if gain > best_gain:
                best_gain = gain
                best_i = i

        if best_i is None:
            break

        selected.append(sentences[best_i])
        selected_idx.add(best_i)
        current_rouge += best_gain

    labels = torch.zeros(len(sentences))
    for i in selected_idx:
        labels[i] = 1.0

    return labels.unsqueeze(0)  # [1, num_sentences]


def train():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    model = BertSumExt().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_docs = json.load(f)

    model.train()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for doc in tqdm(train_docs):
            gold_summary = doc["source"]

            for chunk in doc["chunks"]:
                sentences = chunk["sentences"]
                if len(sentences) == 0:
                    continue

                input_ids, attn_mask, cls_pos = build_bertsum_input(sentences, tokenizer)
                oracle_labels = build_rouge_oracle(
                    sentences, gold_summary, scorer, MAX_ORACLE_SENTENCES
                )

                input_ids = input_ids.to(DEVICE)
                attn_mask = attn_mask.to(DEVICE)
                cls_pos = cls_pos.to(DEVICE)
                oracle_labels = oracle_labels.to(DEVICE)

                logits = model(input_ids, attn_mask, cls_pos)
                loss = criterion(logits, oracle_labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    # ---- inference on validation set (baseline output) ----
    model.eval()
    results = []

    with open(VAL_JSON, "r", encoding="utf-8") as f:
        val_docs = json.load(f)

    with torch.no_grad():
        for doc in tqdm(val_docs):
            all_scored_sentences = []

            for chunk in doc["chunks"]:
                sentences = chunk["sentences"]
                if len(sentences) == 0:
                    continue

                input_ids, attn_mask, cls_pos = build_bertsum_input(sentences, tokenizer)
                logits = model(
                    input_ids.to(DEVICE),
                    attn_mask.to(DEVICE),
                    cls_pos.to(DEVICE)
                )

                scores = logits.squeeze(0).cpu().tolist()
                for s, score in zip(sentences, scores):
                    all_scored_sentences.append((s, score))

            all_scored_sentences.sort(key=lambda x: x[1], reverse=True)
            summary = " ".join([s for s, _ in all_scored_sentences[:MAX_ORACLE_SENTENCES]])

            results.append({
                "doc_id": doc["doc_id"],
                "summary": summary
            })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved BERTSUM baseline to {OUTPUT_JSON}")


if __name__ == "__main__":
    train()