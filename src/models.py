"""
Machine Learning and NLP embedding models and similarity calculations for ResumeIQ.
"""

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask):
    """
    Compute mean pooling of token embeddings, taking attention mask into account.

    Args:
        model_output: Raw model output tuple where first element contains token embeddings.
        attention_mask: Attention mask tensor indicating non-padding tokens.

    Returns:
        Pooled sentence embeddings tensor.
    """
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@st.cache_resource
def load_bert_model_and_tokenizer(model_name: str = "sentence-transformers/bert-base-nli-mean-tokens"):
    """
    Load and cache pretrained Hugging Face tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


@st.cache_resource
def get_HF_embeddings(sentences):
    """
    Generate BERT-based semantic sentence embeddings for given text(s).

    Args:
        sentences: Text string or list of text strings.

    Returns:
        Tensor of pooled embeddings.
    """
    tokenizer, model = load_bert_model_and_tokenizer()

    # Tokenize sentences
    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return embeddings


@st.cache_data
def get_doc2vec_embeddings(JD, text_resume):
    """
    Generate Doc2Vec embeddings for Job Description and resumes.

    Args:
        JD: Job description text string.
        text_resume: List of resume text strings.

    Returns:
        Tuple of (JD_embeddings, resume_embeddings).
    """
    nltk.download("punkt", quiet=True)
    data = [JD]
    resume_embeddings = []

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

    model = Doc2Vec(vector_size=512, min_count=3, epochs=80)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=80)

    if hasattr(model, "dv"):
        jd_vec = model.dv["0"]
    else:
        jd_vec = model.docvecs["0"]
    JD_embeddings = np.transpose(jd_vec.reshape(-1, 1))

    for i in text_resume:
        text = word_tokenize(i.lower())
        embeddings = model.infer_vector(text)
        resume_embeddings.append(np.transpose(embeddings.reshape(-1, 1)))

    return (JD_embeddings, resume_embeddings)


def cosine(embeddings1, embeddings2):
    """
    Calculate pairwise cosine similarity between embeddings.

    Args:
        embeddings1: List/tensor of resume embeddings.
        embeddings2: Job description embedding tensor/array.

    Returns:
        List of match percentage strings rounded to two decimals.
    """
    score_list = []

    # Prepare embedding 2 as 2D numpy array
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
        score_list.append(str(score_val))

    return score_list
