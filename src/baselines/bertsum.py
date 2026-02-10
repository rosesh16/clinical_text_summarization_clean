import torch
import torch.nn as nn
from transformers import BertModel


class BertSumExt(nn.Module):
    """
    BERTSUM Extractive Baseline.
    Scores each sentence using its [CLS] embedding.
    """

    def __init__(self, pretrained_model="bert-base-uncased", hidden_size=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, cls_positions):
        """
        Args:
            input_ids:      [B, T]
            attention_mask: [B, T]
            cls_positions:  [B, N]

        Returns:
            sentence_scores: [B, N]
        """

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state  # [B, T, H]

        batch_size, num_sentences = cls_positions.size()

        cls_embeddings = []
        for b in range(batch_size):
            cls_embeddings.append(hidden_states[b, cls_positions[b]])

        cls_embeddings = torch.stack(cls_embeddings)  # [B, N, H]

        logits = self.classifier(cls_embeddings).squeeze(-1)  # [B, N]

        return logits