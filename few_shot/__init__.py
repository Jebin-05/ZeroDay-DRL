# Few-shot learning module initialization
from .prototypical_network import PrototypicalNetwork, EmbeddingNetwork
from .meta_learner import MetaLearner

__all__ = ['PrototypicalNetwork', 'EmbeddingNetwork', 'MetaLearner']
