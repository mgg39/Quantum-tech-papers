"""
graph_visualization.py

Author: Ricard Santiago Raigada García
Date: 15-02-2025

This module contains functions for rendering graph visualizations of quantum technology research papers in a Streamlit application.
It includes options for visualizing author-paper and keyword-paper relationships.

Functions:
    render_graph_page(train_df, texts) -> None:
        Renders the graph visualization page in the Streamlit application.

    plot_author_paper_graph(df, texts) -> None:
        Plots a graph showing the relationship between authors and papers.

    plot_keyword_paper_graph(df, texts) -> None:
        Plots a graph showing the relationship between keywords and papers.

    plot_graph(G, texts) -> None:
        Plots the graph using Plotly.

    show_legend_paper_author(texts) -> None:
        Displays the legend for the author-paper graph.

    show_legend_paper_keyword(texts) -> None:
        Displays the legend for the keyword-paper graph.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx


def render_graph_page(train_df, texts) -> None:
    """
    Renders the graph visualization page in the Streamlit application.

    Parameters:
    train_df (pd.DataFrame): DataFrame containing the training data.
    texts (dict): Dictionary containing text labels for various fields.

    Returns:
    None
    """
    st.title(texts['graph']['title'])
    st.write(texts['graph']['description'])

    graph_type = st.radio(
        texts['graph']['select_graph_type'],
        ["Author-Paper", "Keyword-Paper"],
        index=0,
        horizontal=True
    )

    if st.button(texts['graph']['generate_graph']):
        if graph_type == "Author-Paper":
            show_legend_paper_author(texts)
            plot_author_paper_graph(train_df, texts)

        elif graph_type == "Keyword-Paper":
            show_legend_paper_keyword(texts)
            plot_keyword_paper_graph(train_df, texts)


def plot_author_paper_graph(df, texts) -> None:
    """
    Plots a graph showing the relationship between authors and papers.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    texts (dict): Dictionary containing text labels for various fields.

    Returns:
    None
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        paper = row['Title']
        paper_details = row.to_dict()

        authors = str(row['Authors']).split(', ')
        G.add_node(paper, type='paper', label=paper, details=paper_details)

        for author in authors:
            if author.strip():
                G.add_node(author, type='author', label=author)
                G.add_edge(author, paper)

    plot_graph(G, texts)


def plot_keyword_paper_graph(df, texts) -> None:
    """
    Plots a graph showing the relationship between keywords and papers.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    texts (dict): Dictionary containing text labels for various fields.

    Returns:
    None
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        paper = row['Title']
        paper_details = row.to_dict()

        keywords = str(row['Keywords']).split(', ')
        G.add_node(paper, type='paper', label=paper, details=paper_details)

        for keyword in keywords:
            keyword = keyword.strip()
            if keyword:
                G.add_node(keyword, type='keyword', label=keyword)
                G.add_edge(keyword, paper)

    plot_graph(G, texts)


def plot_graph(G, texts) -> None:
    """
    Plots the graph using Plotly.

    Parameters:
    G (networkx.Graph): The graph to be plotted.
    texts (dict): Dictionary containing text labels for various fields.

    Returns:
    None
    """
    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y, node_text, node_hover, node_color = [], [], [], [], []

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(data.get('label', node))

        if data.get('type') == 'paper':
            node_hover.append(
                f"{data['label']}<br><b>{texts['paper']['authors']}:</b> {data['details'].get('Authors', 'N/A')}<br>"
                f"<b>{texts['paper']['journal']}:</b> {data['details'].get('Journal', 'N/A')}<br>"
                f"<b>{texts['paper']['keywords']}:</b> {data['details'].get('Keywords', 'N/A')}"
            )
            node_color.append('#1f77b4')
        elif data.get('type') == 'author':
            node_hover.append(f"{data['label']} (Author)")
            node_color.append('#2ca02c')
        elif data.get('type') == 'keyword':
            node_hover.append(f"{data['label']} (Keyword)")
            node_color.append('#ff7f0e')

    fig = go.Figure(
        data=[
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(
                    width=0.5,
                    color='#888'),
                hoverinfo='none'),
            go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                text=node_text,
                textposition='top center',
                marker=dict(
                    size=12,
                    color=node_color,
                    line=dict(
                        width=0.5,
                        color='black')),
                hoverinfo='text',
                hovertext=node_hover,
            )])

    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)


def show_legend_paper_author(texts) -> None:
    """
    Displays the legend for the author-paper graph.

    Parameters:
    texts (dict): Dictionary containing text labels for various fields.

    Returns:
    None
    """
    st.markdown(
        f"""
        #### {texts['graph']['legend_title']}
        - <span style="color:#1f77b4;">⬤ {texts['graph']['legend_paper']}</span>
        - <span style="color:#2ca02c;">⬤ {texts['graph']['legend_author']}</span>
        """,
        unsafe_allow_html=True
    )


def show_legend_paper_keyword(texts) -> None:
    """
    Displays the legend for the keyword-paper graph.

    Parameters:
    texts (dict): Dictionary containing text labels for various fields.

    Returns:
    None
    """
    st.markdown(
        f"""
        #### {texts['graph']['legend_title']}
        - <span style="color:#1f77b4;">⬤ {texts['graph']['legend_paper']}</span>
        - <span style="color:#ff7f0e;">⬤ {texts['graph']['legend_keyword']}</span>
        """,
        unsafe_allow_html=True
    )
