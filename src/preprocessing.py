"""
preprocessing.py

Author: Ricard Santiago Raigada García
Date: 15-02-2025

This module contains functions for preprocessing text data, including Named Entity Recognition (NER) and n-gram generation,
to enhance the quality of research paper titles in a dataset.

Functions:
    ner_and_join_names(text) -> str:
        Detects named entities (PERSON, ORG) and joins multi-word names with underscores. Lowercases other tokens.

    generate_ngrams(df) -> pd.DataFrame:
        Generates bigrams and trigrams from the 'Title' column of the DataFrame.

    apply_ner(df) -> pd.DataFrame:
        Applies Named Entity Recognition (NER) and multi-word entity joining to the 'title_clean' column.
"""
import spacy
from gensim.models import Phrases
from gensim.models.phrases import Phraser


nlp = spacy.load("en_core_web_sm")


def ner_and_join_names(text) -> str:
    """
    Detect named entities (PERSON, ORG) and join multi-word names with underscores.
    Lowercase other tokens.

    Example: "Isaac Chuang" -> "Isaac_Chuang"
             "Fault-tolerant circuits" -> "fault-tolerant circuits"

    Parameters:
    text (str): Input text (title).

    Returns:
    str: Cleaned string with multi-word entities joined and text normalized.
    """
    doc = nlp(text)
    tokens = []

    for ent in doc.ents:
        if ent.label_ in {'PERSON', 'ORG'}:
            tokens.append(ent.text.replace(' ', '_'))

    for token in doc:
        if token.text not in [e.text for e in doc.ents]:
            tokens.append(token.text.lower())

    return ' '.join(tokens)


def generate_ngrams(df) -> pd.DataFrame:
    """
    Generate bigrams and trigrams from the 'Title' column of the DataFrame.
    This captures common multi-word expressions like "quantum computing".

    Parameters:
    df (pd.DataFrame): DataFrame with 'Title' column.

    Returns:
    pd.DataFrame: DataFrame with 'title_clean' column containing processed n-grams.
    """
    sentences = [title.lower().split() for title in df['Title']]

    bigram_model = Phrases(sentences, min_count=2, threshold=3)
    trigram_model = Phrases(bigram_model[sentences], threshold=3)

    bigram_phraser = Phraser(bigram_model)
    trigram_phraser = Phraser(trigram_model)


    df['title_clean'] = [
        ' '.join(trigram_phraser[bigram_phraser[sent]]) for sent in sentences]

    return df


def apply_ner(df) -> pd.DataFrame:
    """
    Apply Named Entity Recognition (NER) and multi-word entity joining to 'title_clean' column.

    Parameters:
    df (pd.DataFrame): DataFrame with 'title_clean' column.

    Returns:
    pd.DataFrame: DataFrame with NER-processed 'title_clean' column.
    """
    df['title_clean'] = df['title_clean'].apply(ner_and_join_names)
    return df
