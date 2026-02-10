import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download("punkt")

class FactVerifier:
    def __init__(self, model_name="roberta-large-mnli", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # MNLI label mapping: [contradiction, neutral, entailment]
        self.ENTAIL_IDX = 2

    def verify_claim(self, claim, evidence_sents):
        """
        Returns max entailment probability over evidence sentences
        """
        max_entail = 0.0

        for ev in evidence_sents:
            inputs = self.tokenizer(
                ev,
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                entail_prob = probs[0, self.ENTAIL_IDX].item()

            max_entail = max(max_entail, entail_prob)

        return max_entail

    def verify_summary(self, summary, evidence_sents, threshold=0.6):
        """
        Returns:
        - list of (claim, entail_score, is_supported)
        - overall factual confidence (mean entailment)
        """
        claims = nltk.sent_tokenize(summary)

        results = []
        scores = []

        for c in claims:
            score = self.verify_claim(c, evidence_sents)
            supported = score >= threshold
            results.append((c, score, supported))
            scores.append(score)

        overall_conf = sum(scores) / max(len(scores), 1)
        return results, overall_conf