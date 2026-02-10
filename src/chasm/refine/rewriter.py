import torch
from transformers import BartTokenizer, BartForConditionalGeneration

class ClaimRewriter:
    """
    Rewrites unsupported claims using only provided evidence.
    """

    def __init__(self, model_name="facebook/bart-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rewrite(self, claim, evidence):
        """
        claim: str
        evidence: List[str]
        """
        prompt = (
            "Rewrite the following clinical statement using only the evidence below. "
            "Do not add new information.\n\n"
            f"Evidence:\n{' '.join(evidence)}\n\n"
            f"Statement:\n{claim}"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                length_penalty=1.5,
                early_stopping=True
            )

        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)