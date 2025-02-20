"""
streamlit_app.py

Author: Ricard Santiago Raigada García
Date: 15-02-2025

This module contains the Streamlit application for searching and filtering quantum technology research papers.
The application allows users to search for papers based on various filters such as categories and authors,
and displays the search results with detailed information about each paper.

Functions:
    render_paper(row, texts) -> None:
        Renders a research paper's details in a Streamlit container.

    run_streamlit_app() -> None:
        Runs the Streamlit application for searching and filtering quantum technology papers.
"""
import streamlit as st
import pandas as pd
from src.inference import load_embeddings_and_model, search_papers_exact_boost
from src.config_loader import load_config, load_texts
from app.pages.graph_visualization import render_graph_page


def render_paper(row, texts) -> None:
    """
    Renders a research paper's details in a Streamlit container.
    Parameters:
    row (pd.Series): A pandas Series containing the paper's details such as Title, Web, Authors, Journal, Keywords, Category, and Abstract.
    texts (dict): A dictionary containing text labels for various fields such as 'paper', 'view_paper', 'authors', 'journal', 'keywords', 'category', and 'show_abstract'.
    The function displays the paper's title, a link to view the paper, and additional details like authors, journal, keywords, category, and abstract if they are available.
    """

    with st.container():
        st.markdown(f"### {row['Title']}")

        st.markdown(f"[{texts['paper']['view_paper']}]({row['Web']})")

        col1, col2, col3 = st.columns([1, 1, 1])

        if pd.notna(row['Authors']) and row['Authors'] != 'Unknown':
            col1.write(f"**{texts['paper']['authors']}:** {row['Authors']}")

        if pd.notna(row['Journal']) and row['Journal'] != 'Unknown':
            col2.write(f"**{texts['paper']['journal']}:** {row['Journal']}")

        if pd.notna(row['Keywords']) and row['Keywords'] != 'Unknown':
            col3.write(f"**{texts['paper']['keywords']}:** {row['Keywords']}")

        if pd.notna(row['Category']):
            st.write(f"**{texts['paper']['category']}:** {row['Category']}")

        if pd.notna(row['Abstract']) and row['Abstract'] != 'Unknown':
            with st.expander(texts['paper']['show_abstract']):
                st.write(row['Abstract'])

        st.divider()


def run_streamlit_app() -> None:
    """
    Runs the Streamlit application for searching and filtering quantum technology papers.
    This function loads the necessary configuration and text resources, sets up the Streamlit
    interface with filters for categories and authors, and provides a search functionality
    for querying papers. The results are displayed based on the selected filters and search query.
    The function performs the following steps:
    1. Loads configuration and text resources.
    2. Loads data including training dataframe, embeddings, model, vectorizer, and TF-IDF matrix.
    3. Sets up the Streamlit interface with two columns for category and author filters.
    4. Filters the dataframe based on selected category and authors.
    5. Provides a search input for querying papers.
    6. Displays the search results or filtered papers.
    7. Displays the end of the list message.
    Returns:
        None
    """
    config = load_config()
    texts = load_texts()
    st.set_page_config(
        page_title="Quantum Tech Papers",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    @st.cache_resource
    def load_data():
        train_df, embeddings, model, vectorizer, X_tfidf = load_embeddings_and_model(
            config)
        return train_df, embeddings, model, vectorizer, X_tfidf

    train_df, embeddings, model, vectorizer, X_tfidf = load_data()

    # Sidebar navigation
    st.sidebar.title(texts["app"]["sidebar_title"])
    page = st.sidebar.radio(
        texts["app"]["sidebar_navigation"],
        [texts["sidebar"]["search_page"], texts["sidebar"]["graph_page"]]
    )

    if page == texts["sidebar"]["search_page"]:
        st.title(texts["app"]["home_page_title"])
        st.subheader(texts["app"]["home_about_title"])

        st.write(texts["app"]["home_about_text_1"])
        st.write(texts["app"]["home_about_text_2"])

        st.markdown(
            f'<span style="color:#1f77b4;">{texts["app"]["home_about_topics"]}</span>',
            unsafe_allow_html=True
        )
        st.write(texts["app"]["home_about_text_3"])
        st.write(
            """
            If you would like to add any paper to this repository please feel free to create a pull request &/or reach out to me via linkedin or email.

            Linkedin : [https://www.linkedin.com/in/maria-gragera-garces/](https://www.linkedin.com/in/maria-gragera-garces/)

            Email : [m.gragera.garces@gmail.com](mailto:m.gragera.garces@gmail.com)
            """
        )
        st.divider()

    if page == texts["sidebar"]["search_page"]:
        col1, col2 = st.columns([2, 2])

        with col1:
            categories = train_df['Category'].dropna().unique().tolist()
            categories.insert(0, texts["filters"]["all"])
            category_selected = st.selectbox(
                texts["filters"]["select_label"], categories)

        with col2:
            all_authors = (
                train_df['Authors'].dropna()
                .str.split(', ')
                .explode()
                .unique()
                .tolist()
            )
            all_authors.sort()
            selected_authors = st.multiselect(
                texts["filters"]["select_authors"], all_authors)

        query = st.text_input(texts["search"]["input_placeholder"])

        if category_selected != texts["filters"]["all"]:
            df_filtered = train_df[train_df['Category'] == category_selected]
        else:
            df_filtered = train_df

        if selected_authors:
            df_filtered = df_filtered[df_filtered['Authors'].apply(
                lambda authors: any(author in str(authors) for author in selected_authors))]

        if query:
            st.subheader(texts["search"]["results_title"].format(query=query))
            results = search_papers_exact_boost(
                query, train_df, embeddings, model, top_k=5, boost_weight=0.5
            )

            for _, row in results.iterrows():
                render_paper(row, texts)

        else:
            st.subheader(
                texts["filters"]["showing_category"].format(
                    category=category_selected))
            st.write(
                texts["filters"]["displaying_n_papers"].format(
                    count=len(df_filtered)))

            for _, row in df_filtered.iterrows():
                render_paper(row, texts)

        st.write(texts["app"]["end_of_list"])

    elif page == texts["sidebar"]["graph_page"]:
        render_graph_page(train_df, texts)
