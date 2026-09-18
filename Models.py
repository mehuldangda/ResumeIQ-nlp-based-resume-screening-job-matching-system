"""
Legacy entrypoint for backward compatibility.
Redirects to src.models.
"""

from src.models import (
    cosine,
    get_HF_embeddings,
    get_doc2vec_embeddings,
    load_bert_model_and_tokenizer,
    mean_pooling,
)

__all__ = [
    "mean_pooling",
    "load_bert_model_and_tokenizer",
    "get_HF_embeddings",
    "get_doc2vec_embeddings",
    "cosine",
]
