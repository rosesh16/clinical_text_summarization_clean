import torch
import torch.nn as nn
from transformers import BertModel


class HiBERT(nn.Module):
    """
    HiBERT-lite: hierarchical extractive model
    Sentence -> Chunk -> Document (implicit)
    """

    def __init__(self, pretrained="bert-base-uncased", hidden=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.sent_fc = nn.Linear(hidden * 2, 1)

    def forward(self, input_ids, attention_mask, cls_positions, chunk_ids):
        """
        input_ids: [B, T]
        attention_mask: [B, T]
        cls_positions: [B, N]       positions of sentence [CLS]
        chunk_ids: [B, N]           chunk index for each sentence
        """

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden = outputs.last_hidden_state  # [B, T, H]
        B, N = cls_positions.shape

        sent_embs = []
        for b in range(B):
            sent_embs.append(hidden[b, cls_positions[b]])
        sent_embs = torch.stack(sent_embs)  # [B, N, H]

        # ---- chunk representations (mean pooling) ----
        chunk_embs = []
        for b in range(B):
            chunks = {}
            for i in range(N):
                cid = int(chunk_ids[b, i])
                chunks.setdefault(cid, []).append(sent_embs[b, i])

            chunk_mean = {
                cid: torch.stack(v).mean(dim=0)
                for cid, v in chunks.items()
            }
            chunk_embs.append(chunk_mean)

        # ---- sentence + chunk context ----
        final_embs = []
        for b in range(B):
            embs = []
            for i in range(N):
                cid = int(chunk_ids[b, i])
                embs.append(
                    torch.cat([sent_embs[b, i], chunk_embs[b][cid]], dim=-1)
                )
            final_embs.append(torch.stack(embs))

        final_embs = torch.stack(final_embs)  # [B, N, 2H]
        logits = self.sent_fc(final_embs).squeeze(-1)  # [B, N]

        return logits