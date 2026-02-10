import numpy as np
from collections import defaultdict


class HierarchicalRanker:
    """
    Refines sentence salience using chunk-level structure.
    """

    def __init__(self, max_per_chunk=1):
        self.max_per_chunk = max_per_chunk

    def rerank(self, sentences, salience_scores, chunk_ids, top_k):
        """
        sentences: List[str]
        salience_scores: np.array [N]
        chunk_ids: List[int] (sentence → chunk mapping)
        """

        chunk_to_sents = defaultdict(list)
        for i, cid in enumerate(chunk_ids):
            chunk_to_sents[cid].append((i, salience_scores[i]))

        # pick top sentences per chunk
        selected = []
        for cid, items in chunk_to_sents.items():
            items = sorted(items, key=lambda x: x[1], reverse=True)
            selected.extend(items[: self.max_per_chunk])

        # global re-ranking
        selected = sorted(selected, key=lambda x: x[1], reverse=True)
        final_idx = [i for i, _ in selected[:top_k]]

        return [sentences[i] for i in sorted(final_idx)]