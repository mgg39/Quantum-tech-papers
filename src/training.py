"""
training.py

Author: Ricard Santiago Raigada García
Date: 15-02-2025

This module contains functions for training and saving models and embeddings for quantum technology research papers.

Functions:
    train_tfidf(train_df, stop_words=None) -> tuple:
        Trains and fits a TF-IDF vectorizer.

    train_sbert_and_embed(train_df, model_name='all-MiniLM-L6-v2') -> tuple:
        Trains SBERT and creates embeddings.

    save_artifacts(vectorizer, X_tfidf, sbert_model, sbert_embeddings, train_df, config) -> None:
        Saves trained models, embeddings, and preprocessed data.

    main(config) -> None:
        Main function to execute the training pipeline.
"""
import os
import pickle
import pandas as pd
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.preprocessing import generate_ngrams, apply_ner


def train_tfidf(train_df, stop_words=None) -> tuple:
    """
    Train and fit a TF-IDF vectorizer.

    Parameters:
    train_df (pd.DataFrame): DataFrame containing the training data with 'title_clean' column.
    stop_words (set): Set of stop words to exclude from the TF-IDF vectorizer. Default is None.

    Returns:
    tuple: A tuple containing the fitted TF-IDF vectorizer and the TF-IDF matrix.
    """
    if stop_words is None:
        stop_words = {
            'quantum',
            'information',
            'computing',
            'system',
            'systems',
            'using',
            'based',
            'study',
            'approach'}

    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        max_df=0.9,
        min_df=2,
        ngram_range=(
            1,
            2))
    X_tfidf = vectorizer.fit_transform(train_df['title_clean'])

    return vectorizer, X_tfidf


def train_sbert_and_embed(train_df, model_name='all-MiniLM-L6-v2') -> tuple:
    """
    Train SBERT and create embeddings.

    Parameters:
    train_df (pd.DataFrame): DataFrame containing the training data with 'title_clean' column.
    model_name (str): Name of the SBERT model to use. Default is 'all-MiniLM-L6-v2'.

    Returns:
    tuple: A tuple containing the SBERT model and the embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        train_df['title_clean'].tolist(),
        show_progress_bar=True)
    return model, embeddings


def save_artifacts(
        vectorizer,
        X_tfidf,
        sbert_model,
        sbert_embeddings,
        train_df,
        config) -> None:
    """
    Save trained models, embeddings, and preprocessed data.

    Parameters:
    vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    X_tfidf (scipy.sparse.csr.csr_matrix): The TF-IDF matrix.
    sbert_model (SentenceTransformer): The trained SBERT model.
    sbert_embeddings (numpy.ndarray): The SBERT embeddings.
    train_df (pd.DataFrame): The DataFrame containing the training data with 'title_clean' column.
    config (dict): Configuration dictionary containing paths for saving artifacts.

    Returns:
    None
    """
    output_dir = config['paths']['output_dir']

    os.makedirs(output_dir, exist_ok=True)

    with open(config['paths']['tfidf_vectorizer'], 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(config['paths']['tfidf_matrix'], 'wb') as f:
        pickle.dump(X_tfidf, f)

    with open(config['paths']['sbert_embeddings'], 'wb') as f:
        pickle.dump(sbert_embeddings, f)

    sbert_model.save(config['paths']['sbert_model'])

    train_df.to_csv(config['paths']['train_data_with_clean'], index=False)

    print(f"Artifacts saved in '{output_dir}'")


def main(config) -> None:
    """
    Main function to execute the training pipeline.

    Parameters:
    config (dict): Configuration dictionary containing paths for input data and saving artifacts.

    Returns:
    None
    """
    df = pd.read_csv(config['paths']['enriched_csv'])

    train_df = generate_ngrams(df)
    train_df = apply_ner(train_df)

    vectorizer, X_tfidf = train_tfidf(train_df)

    sbert_model, sbert_embeddings = train_sbert_and_embed(train_df)

    save_artifacts(
        vectorizer,
        X_tfidf,
        sbert_model,
        sbert_embeddings,
        train_df,
        config)


if __name__ == '__main__':
    main()
