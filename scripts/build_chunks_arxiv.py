import json
import nltk
from pathlib import Path

nltk.download("punkt")

INPUT_PATH = Path("data/processed/arxiv/arxiv_clean.json")
OUTPUT_PATH = Path("data/processed/arxiv/arxiv_val.json")

SENTS_PER_CHUNK = 5

def build_chunks():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    processed_docs = []

    for doc in documents:
        sentences = nltk.sent_tokenize(doc["text"])

        chunks = []
        for i in range(0, len(sentences), SENTS_PER_CHUNK):
            chunks.append({
                "sentences": sentences[i:i + SENTS_PER_CHUNK]
            })

        processed_docs.append({
            "doc_id": doc["doc_id"],
            "chunks": chunks,
            "summary": doc["summary"]
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, indent=2)

    print(f"[Chunk Build] Documents processed: {len(processed_docs)}")
    print(f"[Chunk Build] Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_chunks()