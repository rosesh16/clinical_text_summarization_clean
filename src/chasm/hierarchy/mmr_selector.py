import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class MultiDocMMRSelector:
    """
    Multi-document redundancy-aware sentence selector using MMR.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)

    def select(
        self,
        sentences,
        salience_scores,
        doc_ids,
        top_k=8,
        lambda_param=0.7,
        max_per_doc=None
    ):

        N = len(sentences)
        embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
        similarity_matrix = cosine_similarity(embeddings)

        selected = []
        selected_docs_count = {}

        salience = (salience_scores - salience_scores.min()) / (
            salience_scores.max() - salience_scores.min() + 1e-8
        )

        while len(selected) < min(top_k, N):

            mmr_scores = []

            for i in range(N):
                if i in selected:
                    mmr_scores.append(-np.inf)
                    continue

                if max_per_doc is not None:
                    doc_id = doc_ids[i]
                    if selected_docs_count.get(doc_id, 0) >= max_per_doc:
                        mmr_scores.append(-np.inf)
                        continue

                if not selected:
                    redundancy_penalty = 0
                else:
                    redundancy_penalty = max(
                        similarity_matrix[i][j] for j in selected
                    )

                mmr_score = (
                    lambda_param * salience[i]
                    - (1 - lambda_param) * redundancy_penalty
                )

                mmr_scores.append(mmr_score)

            best_idx = np.argmax(mmr_scores)

            if mmr_scores[best_idx] == -np.inf:
                break

            selected.append(best_idx)

            doc_id = doc_ids[best_idx]
            selected_docs_count[doc_id] = (
                selected_docs_count.get(doc_id, 0) + 1
            )

        selected = sorted(selected)
        return [sentences[i] for i in selected], selected