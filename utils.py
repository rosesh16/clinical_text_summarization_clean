import hashlib
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Iterable, List

import nltk
from pypdf import PdfReader

from src.exception import CustomException
from src.logger import logging


DEFAULT_CACHE_DIR = Path("cache")


def save_object(file_path: str, obj):
    """
    Save a Python object with pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Object saved successfully at %s", file_path)
    except Exception as e:
        logging.error("Error occurred while saving object")
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Load a Python object saved with pickle.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info("Object loaded successfully from %s", file_path)
        return obj
    except Exception as e:
        logging.error("Error occurred while loading object")
        raise CustomException(e, sys)


def create_directories(paths: list):
    """
    Create directories if they do not already exist.
    """
    try:
        for path in paths:
            os.makedirs(path, exist_ok=True)
            logging.info("Directory created or already exists: %s", path)
    except Exception as e:
        logging.error("Error occurred while creating directories")
        raise CustomException(e, sys)


def ensure_nltk_resource(resource_name: str, download_name: str | None = None):
    """
    Download an NLTK resource only when it is missing.
    """
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(download_name or resource_name.split("/")[-1], quiet=True)


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from an uploaded PDF file.
    """
    try:
        reader = PdfReader(pdf_file)
        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            cleaned_page = normalize_whitespace(page_text)
            if cleaned_page:
                pages.append(cleaned_page)

        text = "\n".join(pages).strip()
        logging.info("Extracted %s characters from PDF", len(text))
        return text
    except Exception as e:
        logging.error("Failed to extract text from PDF")
        raise CustomException(e, sys)


def normalize_whitespace(text: str) -> str:
    """
    Collapse repeated whitespace while keeping the text readable.
    """
    return re.sub(r"\s+", " ", text).strip()


def sentence_split(text: str) -> List[str]:
    """
    Split text into sentences using NLTK, with a regex fallback.
    """
    cleaned_text = normalize_whitespace(text)
    if not cleaned_text:
        return []

    ensure_nltk_resource("tokenizers/punkt")

    try:
        sentences = nltk.sent_tokenize(cleaned_text)
    except LookupError:
        sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_text(text: str, min_words: int = 300, max_words: int = 500) -> List[str]:
    """
    Build sentence-aware chunks with a soft target between min_words and max_words.
    """
    sentences = sentence_split(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current_sentences and current_words >= min_words:
            would_exceed = current_words + sentence_words > max_words
            is_very_long_sentence = sentence_words > max_words
            if would_exceed or is_very_long_sentence:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_words = 0

        current_sentences.append(sentence)
        current_words += sentence_words

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    logging.info("Created %s semantic chunks", len(chunks))
    return chunks


def hash_texts(texts: Iterable[str], prefix: str = "") -> str:
    """
    Create a stable hash for a sequence of texts.
    """
    digest = hashlib.sha256()
    if prefix:
        digest.update(prefix.encode("utf-8"))

    for text in texts:
        digest.update(text.encode("utf-8", errors="ignore"))
        digest.update(b"||")

    return digest.hexdigest()


def get_cache_path(cache_dir: str | Path, cache_name: str) -> Path:
    """
    Build a cache file path and ensure the cache directory exists.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / cache_name


def load_cached_data(cache_path: str | Path):
    """
    Load cached data when it exists; otherwise return None.
    """
    path = Path(cache_path)
    if not path.exists():
        return None

    try:
        return load_object(str(path))
    except Exception:
        logging.warning("Cache load failed for %s. Recomputing.", path)
        return None


def save_cached_data(cache_path: str | Path, data):
    """
    Save cached data to disk.
    """
    save_object(str(cache_path), data)

