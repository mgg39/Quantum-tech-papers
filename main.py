"""
main.py

Author: Ricard Santiago Raigada García
Date: 15-02-2025

This module serves as the entry point for the Streamlit application. It imports and runs the necessary functions
to start the Streamlit app and to enrich the dataset of quantum technology research papers incrementally.

Functions:
    main() -> None:
        Runs the Streamlit application.
"""
import streamlit as st
from app.streamlit_app import run_streamlit_app
from src.enrich_papers_incremental import enrich_incrementally

if __name__ == '__main__':
    run_streamlit_app()
