"""
enrich_papers_incremental.py

Author: Ricard Santiago Raigada García
Date: 15-02-2025

This module contains functions for extracting metadata from arXiv and CrossRef (DOI) URLs and enriching a dataset of quantum technology research papers incrementally.

Functions:
    extract_arxiv_id(url) -> str:
        Extracts the arXiv ID from a given paper URL.

    extract_doi_from_url(url) -> str:
        Extracts the DOI from a given paper URL.

    get_arxiv_metadata(arxiv_id) -> dict:
        Retrieves paper metadata from the arXiv API.

    get_crossref_metadata(doi) -> dict:
        Retrieves paper metadata from the CrossRef API using a DOI.

    fetch_paper_metadata(url) -> dict:
        Fetches paper metadata by detecting if it is from arXiv or CrossRef (DOI).

    enrich_incrementally(input_csv='data/inputs/papers_data.csv', enriched_csv='data/enriched/papers_data_enriched.csv') -> None:
        Enriches the dataset of quantum technology research papers incrementally by fetching metadata for new papers.
"""
import re
import requests
import pandas as pd
import time
import os


def extract_arxiv_id(url) -> str:
    """
    Extract the arXiv ID from a given paper URL.

    Parameters:
    url (str): Paper URL.

    Returns:
    str: arXiv ID or None if not found.
    """
    match = re.search(r'arxiv\.org/(abs|pdf)/([0-9]+\.[0-9]+)', url)
    if match:
        return match.group(2)
    return None


def extract_doi_from_url(url) -> str:
    """
    Extract the DOI from a given paper URL.

    Parameters:
    url (str): Paper URL.

    Returns:
    str: DOI string or None if not found.
    """
    match = re.search(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', url)
    if match:
        return match.group(0)
    return None


def get_arxiv_metadata(arxiv_id) -> dict:
    """
    Retrieve paper metadata from the arXiv API.

    Parameters:
    arxiv_id (str): arXiv paper identifier.

    Returns:
    dict: Dictionary containing metadata or None if request fails.
    """
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.text)
    ns = {'arxiv': 'http://www.w3.org/2005/Atom'}

    entry = root.find('arxiv:entry', ns)
    if entry is None:
        return None

    title = entry.find('arxiv:title', ns).text.strip()
    abstract = entry.find('arxiv:summary', ns).text.strip()
    authors = ', '.join([author.find(
        'arxiv:name', ns).text for author in entry.findall('arxiv:author', ns)])
    categories = ', '.join([cat.attrib['term']
                           for cat in entry.findall('arxiv:category', ns)])
    keywords = categories if categories else "Unknown"

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": "arXiv",
        "keywords": keywords
    }


def get_crossref_metadata(doi) -> dict:
    """
    Retrieve paper metadata from the CrossRef API using a DOI.

    Parameters:
    doi (str): DOI of the paper.

    Returns:
    dict: Dictionary containing metadata or None if request fails.
    """
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json().get('message', {})
    title = data.get('title', ["Unknown"])[0]
    abstract = data.get('abstract', "Unknown")
    authors = ', '.join([f"{a.get('given', '')} {a.get('family', '')}".strip(
    ) for a in data.get('author', [])])
    journal = data.get('container-title', ["Unknown"])[0]
    keywords = ', '.join(
        data.get(
            'subject',
            [])) if 'subject' in data else "Unknown"

    abstract = re.sub(r'<[^>]*>', '', abstract)

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "keywords": keywords
    }


def fetch_paper_metadata(url):
    """
    Fetch paper metadata by detecting if it is from arXiv or CrossRef (DOI).

    Parameters:
    url (str): Paper URL.

    Returns:
    dict: Metadata dictionary with title, abstract, authors, journal, and keywords.
    """
    arxiv_id = extract_arxiv_id(url)
    doi = extract_doi_from_url(url)

    if arxiv_id:
        metadata = get_arxiv_metadata(arxiv_id)
        if metadata:
            return metadata

    if doi:
        metadata = get_crossref_metadata(doi)
        if metadata:
            return metadata

    return {
        "title": "Unknown",
        "abstract": "Unknown",
        "authors": "Unknown",
        "journal": "Unknown",
        "keywords": "Unknown"
    }


def enrich_incrementally(
        input_csv='data/inputs/papers_data.csv',
        enriched_csv='data/enriched/papers_data_enriched.csv') -> None:
    """
    Enriches the dataset of quantum technology research papers incrementally by fetching metadata for new papers.

    Parameters:
    input_csv (str): Path to the input CSV file containing paper data. Default is 'data/inputs/papers_data.csv'.
    enriched_csv (str): Path to the enriched CSV file to save the enriched data. Default is 'data/enriched/papers_data_enriched.csv'.

    Returns:
    None
    """
    df_new = pd.read_csv(input_csv)

    os.makedirs(os.path.dirname(enriched_csv), exist_ok=True)

    if os.path.exists(enriched_csv):
        df_enriched = pd.read_csv(enriched_csv)
    else:
        df_enriched = pd.DataFrame()

    known_urls = set(df_enriched['Web']) if not df_enriched.empty else set()

    new_rows = []
    for idx, row in df_new.iterrows():
        if row['Web'] not in known_urls:
            print(f"Fetching metadata for {row['Web']}...")
            metadata = fetch_paper_metadata(row['Web'])
            row_data = {
                'Title': row['Title'],
                'Web': row['Web'],
                'Category': row.get('Category', None),
                'Fetched_Title': metadata['title'],
                'Abstract': metadata['abstract'],
                'Authors': metadata['authors'],
                'Journal': metadata['journal'],
                'Keywords': metadata['keywords']
            }
            new_rows.append(row_data)
            # time.sleep(1)

    df_new_entries = pd.DataFrame(new_rows)
    df_final = pd.concat([df_enriched, df_new_entries], ignore_index=True)

    df_final = df_final.sort_values(by=['Fetched_Title'], na_position='last')
    df_final = df_final.drop_duplicates(subset='Web', keep='first')

    df_final.to_csv(enriched_csv, index=False)
    print(f"Enriched CSV saved to {enriched_csv} with {len(df_final)} entries")
