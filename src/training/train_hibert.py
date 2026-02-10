import json
import os
import sys
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW
from rouge_score import rouge_scorer
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.baselines.hibert import HiBERT


TRAIN_JSON = "data/processed/pubmed/pubmed_train.json"
VAL_JSON   = "data/processed/pubmed/pubmed_val.json"
OUTPUT_JSON = "experiments/baseline_results/hibert.json"

MAX_TOKENS = 512
MAX_ORACLE = 3
LR = 2e-5
EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_input(chunks, tokenizer):
    tokens = []
    cls_pos = []
    chunk_ids = []

    for cid, chunk in enumerate(chunks):
        for sent in chunk["sentences"]:
            # record position BEFORE adding tokens
            cls_index = len(tokens)

            sent_tokens = [tokenizer.cls_token] + tokenizer.tokenize(sent) + [tokenizer.sep_token]

            # ---- STOP if this sentence would exceed max length ----
            if cls_index + len(sent_tokens) > MAX_TOKENS:
                break

            cls_pos.append(cls_index)
            chunk_ids.append(cid)
            tokens.extend(sent_tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    return (
        torch.tensor(input_ids).unsqueeze(0),
        torch.tensor(attention_mask).unsqueeze(0),
        torch.tensor(cls_pos).unsqueeze(0),
        torch.tensor(chunk_ids).unsqueeze(0),
    )

def rouge_oracle(sentences, gold, scorer):
    selected = set()
    labels = torch.zeros(len(sentences))
    current = ""

    for _ in range(min(MAX_ORACLE, len(sentences))):
        best, best_i = 0, None
        for i, s in enumerate(sentences):
            if i in selected:
                continue
            score = scorer.score(current + " " + s, gold)["rougeL"].fmeasure
            if score > best:
                best, best_i = score, i
        if best_i is None:
            break
        selected.add(best_i)
        current += " " + sentences[best_i]

    for i in selected:
        labels[i] = 1.0
    return labels.unsqueeze(0)


def train():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    model = HiBERT().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    with open(TRAIN_JSON) as f:
        train_docs = json.load(f)

    model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for doc in tqdm(train_docs):
            sentences = []
            for c in doc["chunks"]:
                sentences.extend(c["sentences"])
            if not sentences:
                continue

            inp = build_input(doc["chunks"], tokenizer)
            # skip if no sentences survived truncation
            if inp[2].numel() == 0:
                continue
            labels = rouge_oracle(sentences, doc["source"], scorer)

            inp = [x.to(DEVICE) for x in inp]
            labels = labels.to(DEVICE)

            logits = model(*inp)
            if logits.size(1) != labels.size(1):
             continue
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # ---- inference on val ----
    model.eval()
    results = []

    with open(VAL_JSON) as f:
        val_docs = json.load(f)

    with torch.no_grad():
        for doc in tqdm(val_docs):
            sentences = []
            for c in doc["chunks"]:
                sentences.extend(c["sentences"])
            if not sentences:
                summary = ""
            else:
                inp = build_input(doc["chunks"], tokenizer)
                if inp[2].numel() == 0:
                  continue
                # number of sentences that survived truncation
                num_valid_sents = inp[2].size(1)

                # truncate sentence list to match model input
                sentences = sentences[:num_valid_sents]

                inp = [x.to(DEVICE) for x in inp]
                scores = model(*inp).squeeze(0).cpu().tolist()

                ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
                summary = " ".join(s for s, _ in ranked[:MAX_ORACLE])

            results.append({
                "doc_id": doc["doc_id"],
                "summary": summary
            })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved HiBERT baseline to {OUTPUT_JSON}")


if __name__ == "__main__":
    train()