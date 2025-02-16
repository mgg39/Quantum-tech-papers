"""
inference.py

Author: Ricard Santiago Raigada García
Date: 16-02-2025

This module contains functions for loading embeddings and models, and for performing semantic and hybrid searches
on quantum technology research papers.

Functions:
    load_embeddings_and_model(config) -> tuple:
        Loads SBERT embeddings, model, training dataframe, TF-IDF vectorizer, and TF-IDF matrix from the specified configuration paths.

    cosine_similarity_numpy(a, b) -> np.ndarray:
        Computes the cosine similarity between a single vector `a` and a matrix `b`.

    search_papers_semantic(query, train_df, train_embeddings, model, top_k=5) -> pd.DataFrame:
        Performs a semantic search using SBERT on the training dataframe and returns the top-k results.

    search_papers_exact_boost(query, train_df, train_embeddings, model, top_k=5, boost_weight=0.5) -> pd.DataFrame:
        Performs a hybrid search combining SBERT semantic similarity with exact match boosting on the training dataframe and returns the top-k results.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd


def load_embeddings_and_model(config) -> tuple:
    """
    Loads SBERT embeddings, model, training dataframe, TF-IDF vectorizer, and TF-IDF matrix from the specified configuration paths.

    Parameters:
    config (dict): A dictionary containing paths to the SBERT embeddings, SBERT model, training data, TF-IDF vectorizer, and TF-IDF matrix.

    Returns:
    tuple: A tuple containing the training dataframe, SBERT embeddings, SBERT model, TF-IDF vectorizer, and TF-IDF matrix.
    """
    with open(config['paths']['sbert_embeddings'], 'rb') as f:
        embeddings = pickle.load(f)

    model = SentenceTransformer(config['paths']['sbert_model'])
    train_df = pd.read_csv(config['paths']['train_data_with_clean'])

    with open(config['paths']['tfidf_vectorizer'], 'rb') as f:
        vectorizer = pickle.load(f)

    with open(config['paths']['tfidf_matrix'], 'rb') as f:
        X_tfidf = pickle.load(f)

    return train_df, embeddings, model, vectorizer, X_tfidf


def cosine_similarity_numpy(a, b):
    """
    Computes the cosine similarity between a single vector `a` and a matrix `b`.

    Parameters:
    a (np.ndarray): A 2D numpy array with shape (1, embedding_dim).
    b (np.ndarray): A 2D numpy array with shape (num_samples, embedding_dim).

    Returns:
    np.ndarray: A 1D array containing cosine similarities between `a` and each row in `b`.
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm.T).flatten()


def search_papers_semantic(
        query,
        train_df,
        train_embeddings,
        model,
        top_k=5) -> pd.DataFrame:
    """
    Performs a semantic search using SBERT on the training dataframe and returns the top-k results.

    Parameters:
    query (str): The search query.
    train_df (pd.DataFrame): The training dataframe containing research papers.
    train_embeddings (np.ndarray): The SBERT embeddings for the training dataframe.
    model (SentenceTransformer): The SBERT model.
    top_k (int): The number of top results to return. Default is 5.

    Returns:
    pd.DataFrame: A dataframe containing the top-k search results with their similarity scores.
    """
    query_embedding = model.encode([query])
    similarities = cosine_similarity_numpy(query_embedding, train_embeddings)
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = train_df.iloc[top_indices][['Title',
                                          'Web',
                                          'Category',
                                          'Abstract',
                                          'Authors',
                                          'Journal',
                                          'Keywords']].copy()
    results['Similarity Score'] = similarities[top_indices]
    return results


def search_papers_exact_boost(
        query,
        train_df,
        train_embeddings,
        model,
        top_k=5,
        boost_weight=0.5) -> pd.DataFrame:
    """
    Performs a hybrid search combining SBERT semantic similarity with exact match boosting on the training dataframe and returns the top-k results.

    Parameters:
    query (str): The search query.
    train_df (pd.DataFrame): The training dataframe containing research papers.
    train_embeddings (np.ndarray): The SBERT embeddings for the training dataframe.
    model (SentenceTransformer): The SBERT model.
    top_k (int): The number of top results to return. Default is 5.
    boost_weight (float): The weight to apply to exact matches in the title. Default is 0.5.

    Returns:
    pd.DataFrame: A dataframe containing the top-k search results with their similarity scores.
    """
    query_embedding = model.encode([query])
    similarities_sbert = cosine_similarity_numpy(
        query_embedding, train_embeddings)

    exact_matches = train_df['Title'].str.contains(
        query, case=False, na=False).astype(float)

    final_score = similarities_sbert + boost_weight * exact_matches

    top_indices = final_score.argsort()[-top_k:][::-1]

    results = train_df.iloc[top_indices][['Title',
                                          'Web',
                                          'Category',
                                          'Abstract',
                                          'Authors',
                                          'Journal',
                                          'Keywords']].copy()
    results['Similarity Score'] = final_score[top_indices]
    return results
