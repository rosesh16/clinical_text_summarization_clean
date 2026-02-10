import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SalienceGraphBuilder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)

    def build_graph(self, sentences):
        """
        sentences: List[str]
        returns: similarity matrix (N x N)
        """
        embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
        sim_matrix = cosine_similarity(embeddings)

        # zero self-loops
        np.fill_diagonal(sim_matrix, 0.0)
        return sim_matrix