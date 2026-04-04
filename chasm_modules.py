from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer

from src.logger import logging
from utils import sentence_split


@dataclass
class VerificationResult:
    final_summary: str
    claim_results: List[dict]
    factual_score: float
    refined: bool


class CHASMSummarizer:
    """
    Multi-stage summarizer used for chunk-level and final aggregation summaries.
    """

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logging.info("Loaded summarization model: %s", model_name)

    def summarize_text(
        self,
        text: str,
        max_input_length: int = 1024,
        max_summary_length: int = 180,
        min_summary_length: int = 40,
    ) -> str:
        """
        Summarize a single text block.
        """
        if not text.strip():
            return ""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_summary_length,
                min_length=min_summary_length,
                num_beams=4,
                length_penalty=1.5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def summarize_chunks(
        self,
        chunks: List[str],
        max_input_length: int = 1024,
        max_summary_length: int = 120,
        min_summary_length: int = 30,
    ) -> List[str]:
        """
        Summarize each chunk independently as the first CHASM stage.
        """
        summaries = []
        for chunk in chunks:
            summaries.append(
                self.summarize_text(
                    text=chunk,
                    max_input_length=max_input_length,
                    max_summary_length=max_summary_length,
                    min_summary_length=min_summary_length,
                )
            )
        return summaries

    def combine_summaries(
        self,
        summaries: List[str],
        max_input_length: int = 1024,
        max_summary_length: int = 220,
        min_summary_length: int = 60,
    ) -> str:
        """
        Combine chunk summaries into a final document summary.
        """
        combined_text = " ".join(summary for summary in summaries if summary.strip())
        return self.summarize_text(
            text=combined_text,
            max_input_length=max_input_length,
            max_summary_length=max_summary_length,
            min_summary_length=min_summary_length,
        )

    def rewrite_with_evidence(
        self,
        summary: str,
        evidence_chunks: List[str],
        max_input_length: int = 1024,
        max_summary_length: int = 220,
        min_summary_length: int = 60,
    ) -> str:
        """
        Refine a summary by constraining generation to the retrieved evidence.
        """
        evidence_text = " ".join(evidence_chunks)
        prompt = (
            "Rewrite the medical summary using only the evidence provided. "
            "Keep the most important clinical findings and avoid unsupported claims.\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Draft Summary:\n{summary}"
        )
        return self.summarize_text(
            text=prompt,
            max_input_length=max_input_length,
            max_summary_length=max_summary_length,
            min_summary_length=min_summary_length,
        )


class FactualVerifier:
    """
    Sentence-level factual verification using an MNLI model.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.entailment_index = 2
        self.contradiction_index = 0
        logging.info("Loaded verifier model: %s", model_name)

    def verify_claim(self, claim: str, evidence: str) -> dict:
        """
        Score a summary sentence against a single evidence text.
        """
        inputs = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        return {
            "entailment": float(probabilities[self.entailment_index].item()),
            "contradiction": float(probabilities[self.contradiction_index].item()),
        }

    def verify_summary(self, summary: str, evidence_chunks: List[str], threshold: float = 0.55) -> List[dict]:
        """
        Verify each sentence in the summary against the retrieved evidence chunks.
        """
        claims = sentence_split(summary)
        results: List[dict] = []

        for claim in claims:
            evidence_scores = [self.verify_claim(claim, evidence) for evidence in evidence_chunks if evidence.strip()]
            if not evidence_scores:
                results.append(
                    {
                        "claim": claim,
                        "best_entailment": 0.0,
                        "best_contradiction": 0.0,
                        "supported": False,
                    }
                )
                continue

            best_score = max(evidence_scores, key=lambda item: item["entailment"])
            results.append(
                {
                    "claim": claim,
                    "best_entailment": best_score["entailment"],
                    "best_contradiction": best_score["contradiction"],
                    "supported": best_score["entailment"] >= threshold
                    and best_score["contradiction"] < best_score["entailment"],
                }
            )

        return results

    def factual_score(self, claim_results: List[dict]) -> float:
        """
        Mean entailment score across summary claims.
        """
        if not claim_results:
            return 0.0
        return sum(item["best_entailment"] for item in claim_results) / len(claim_results)


def verify_and_refine_summary(
    summarizer: CHASMSummarizer,
    verifier: FactualVerifier,
    summary: str,
    evidence_chunks: List[str],
    threshold: float = 0.55,
    max_refinement_attempts: int = 1,
    max_input_length: int = 1024,
    max_summary_length: int = 220,
    min_summary_length: int = 60,
) -> VerificationResult:
    """
    Verify a summary and refine it once when unsupported claims are detected.
    """
    current_summary = summary
    refined = False

    for _ in range(max_refinement_attempts + 1):
        claim_results = verifier.verify_summary(current_summary, evidence_chunks, threshold=threshold)
        if claim_results and all(result["supported"] for result in claim_results):
            return VerificationResult(
                final_summary=current_summary,
                claim_results=claim_results,
                factual_score=verifier.factual_score(claim_results),
                refined=refined,
            )

        if refined or not evidence_chunks:
            break

        current_summary = summarizer.rewrite_with_evidence(
            summary=current_summary,
            evidence_chunks=evidence_chunks,
            max_input_length=max_input_length,
            max_summary_length=max_summary_length,
            min_summary_length=min_summary_length,
        )
        refined = True

    claim_results = verifier.verify_summary(current_summary, evidence_chunks, threshold=threshold)
    return VerificationResult(
        final_summary=current_summary,
        claim_results=claim_results,
        factual_score=verifier.factual_score(claim_results),
        refined=refined,
    )

