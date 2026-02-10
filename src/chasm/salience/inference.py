import torch
from .salience_model import SalienceFusion
from .scorers import normalize, position_prior

class SalienceExtractor:
    def __init__(self, fusion_model: SalienceFusion):
        self.model = fusion_model.eval()

    def extract(self, sentences, textrank_scores, bertsum_scores, k=10):
        pos_scores = position_prior(len(sentences))

        scores = self.model(
            torch.tensor(normalize(textrank_scores)),
            torch.tensor(normalize(bertsum_scores)),
            torch.tensor(normalize(pos_scores)),
        )

        topk = torch.topk(scores, k=min(k, len(sentences))).indices.tolist()
        return [sentences[i] for i in topk], scores.detach().numpy()