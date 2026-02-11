import torch
from transformers import BartTokenizer, BartForConditionalGeneration

class BartGenerator:
    def __init__(self, model_name="facebook/bart-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(self, text, max_len=256, min_len=80):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_len,
                min_length=min_len,
                num_beams=2 ,
                length_penalty=1.0,
                early_stopping=True
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)