import numpy as np

def position_prior(num_sentences):
    """
    Lead bias: earlier sentences are more salient
    """
    return np.array([
        1.0 / (i + 1) for i in range(num_sentences)
    ])

def normalize(x):
    x = np.array(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)