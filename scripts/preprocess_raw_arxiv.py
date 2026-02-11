import json
import re
from pathlib import Path

INPUT_PATH = Path("data/raw/arxiv/arxiv.json")
OUTPUT_PATH = Path("data/processed/arxiv/arxiv_clean.json")


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove LaTeX math placeholders like @xmath12
    text = re.sub(r"@xmath\d+", "", text)

    # Remove citation markers like @xcite
    text = re.sub(r"@xcite", "", text)

    # Remove figure references like [ fig1 ]
    text = re.sub(r"\[\s*fig\d+\s*\]", "", text)

    # Remove numeric citations like [12], [1,2]
    text = re.sub(r"\[[0-9,\s]+\]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove non-printable characters
    text = "".join(ch for ch in text if ch.isprintable())

    return text.strip()

def preprocess():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    cleaned_docs = []

    for doc in documents:
        cleaned_doc = doc.copy()
        cleaned_doc["text"] = clean_text(doc["text"])
        cleaned_doc["summary"] = clean_text(doc["summary"])
        cleaned_docs.append(cleaned_doc)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_docs, f, indent=2, ensure_ascii=False)

    print(f"[Preprocess arXiv] Documents processed: {len(cleaned_docs)}")
    print(f"[Preprocess arXiv] Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess()