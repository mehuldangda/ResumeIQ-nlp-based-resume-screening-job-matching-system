"""
Machine Learning and NLP embedding models and similarity calculations for ResumeIQ.
Provides Hugging Face BERT semantic embeddings, Doc2Vec document vectorization,
and Cosine Similarity evaluation.
"""

from typing import List, Tuple, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def mean_pooling(model_output: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute attention-weighted mean pooling of token embeddings.

    Mean pooling averages all contextual token embeddings produced by BERT,
    masking out padding tokens so they do not skew the document representation.

    Args:
        model_output: Raw model output tuple where the first element contains
                      token embeddings [batch_size, seq_len, hidden_dim].
        attention_mask: Attention mask tensor [batch_size, seq_len] with 1s for
                        valid tokens and 0s for padding.

    Returns:
        torch.Tensor: Pooled sentence embeddings tensor of shape [batch_size, hidden_dim].
    """
    token_embeddings = model_output[0]  # First element contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


@st.cache_resource(show_spinner=False)
def load_bert_model_and_tokenizer(
    model_name: str = "sentence-transformers/bert-base-nli-mean-tokens"
) -> Tuple[AutoTokenizer, AutoModel]:
    """
    Load and cache pretrained Hugging Face tokenizer and transformer model.

    Cached with @st.cache_resource so the model weights (approx 420MB) are loaded
    into memory only once across multiple user sessions.

    Args:
        model_name: Hugging Face model repository identifier.

    Returns:
        Tuple[AutoTokenizer, AutoModel]: Loaded tokenizer and transformer model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # Put model in inference evaluation mode
    return tokenizer, model


def get_HF_embeddings(sentences: Union[str, List[str]]) -> torch.Tensor:
    """
    Generate BERT-based semantic embeddings for one or more text inputs.

    Steps:
    1. Tokenize input text with padding and truncation (max length 512).
    2. Pass tokens through BERT without gradient tracking (torch.no_grad).
    3. Apply attention-weighted mean pooling over token embeddings.

    Args:
        sentences: A single text string or a list of text strings.

    Returns:
        torch.Tensor: Document embedding tensor of shape [num_sentences, 768].
    """
    tokenizer, model = load_bert_model_and_tokenizer()

    if isinstance(sentences, str):
        sentences = [sentences]

    # Tokenize input sequences
    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    # Compute token embeddings without backpropagation gradients
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform mean pooling
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return embeddings


def ensure_nltk_resources() -> None:
    """
    Download required NLTK tokenizer models if not already present.
    Ensures punkt and punkt_tab are available across all NLTK versions.
    """
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


def get_doc2vec_embeddings(
    JD: str,
    text_resume: List[str]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate Doc2Vec dense vector representations for Job Description and resumes.

    Trains a lightweight Doc2Vec model on the fly across the provided texts to
    produce 512-dimensional document vectors.

    Args:
        JD: Job description text string.
        text_resume: List of resume text strings.

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: (JD_embedding of shape [1, 512],
                                             list of resume embeddings each [1, 512]).
    """
    ensure_nltk_resources()

    all_docs = [JD] + list(text_resume)
    tagged_data = [
        TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)])
        for i, doc in enumerate(all_docs)
    ]

    # Initialize and train Doc2Vec
    model = Doc2Vec(vector_size=512, min_count=1, epochs=80)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Infer or retrieve JD vector (tag '0')
    if hasattr(model, "dv"):
        jd_vec = model.dv["0"]
    else:
        jd_vec = model.docvecs["0"]
    jd_embedding = jd_vec.reshape(1, -1)

    # Generate resume vectors
    resume_embeddings = []
    for resume_text in text_resume:
        tokens = word_tokenize(resume_text.lower())
        vec = model.infer_vector(tokens).reshape(1, -1)
        resume_embeddings.append(vec)

    return jd_embedding, resume_embeddings


def cosine(
    embeddings1: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    embeddings2: Union[torch.Tensor, np.ndarray]
) -> List[str]:
    """
    Calculate pairwise cosine similarity between resume embeddings and a job description embedding.

    Formula:
        Cosine Similarity = (A . B) / (||A|| * ||B||)

    Args:
        embeddings1: List or tensor of resume embeddings.
        embeddings2: Job description embedding tensor or array.

    Returns:
        List[str]: List of percentage similarity strings rounded to 2 decimal places.
    """
    score_list = []

    # Convert embedding 2 to 2D numpy array
    if isinstance(embeddings2, torch.Tensor):
        emb2 = embeddings2.detach().cpu().numpy()
    else:
        emb2 = np.asarray(embeddings2)

    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)

    for i in embeddings1:
        if isinstance(i, torch.Tensor):
            emb1 = i.detach().cpu().numpy()
        else:
            emb1 = np.asarray(i)

        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)

        match_percentage = cosine_similarity(emb1, emb2)
        match_percentage = np.round(match_percentage, 4) * 100
        score_val = float(match_percentage[0][0])
        score_list.append(str(round(score_val, 2)))

    return score_list
