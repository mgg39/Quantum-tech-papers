"""
extract_papers_to_csv.py

Author: Ricard Santiago Raigada García
Date: 15-02-2025

This module contains a function for extracting paper data (Title, Web, Category) from a README.md file and saving it to a CSV file.

Functions:
    extract_data_from_readme(input_path, output_path) -> None:
        Extracts paper data (Title, Web, Category) from README.md and saves it to a CSV file.
"""
import re
import csv
import os


def extract_data_from_readme(input_path, output_path) -> None:
    """
    Extract paper data (Title, Web, Category) from README.md and save to CSV.

    Parameters:
    input_path (str): Path to the README.md file.
    output_path (str): Path to the output CSV file.

    Returns:
    None
    """
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    papers = []
    current_category = None

    for line in content:
        category_match = re.match(r"### (.+)", line)
        if category_match:
            current_category = category_match.group(1).strip()

        paper_match = re.match(r"- (.+)\n", line)
        if paper_match:
            title = paper_match.group(1).strip()
            next_line_index = content.index(line) + 1
            if next_line_index < len(content):
                url_line = content[next_line_index].strip()
                if url_line.startswith("http"):
                    papers.append([title, url_line, current_category])

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Title", "Web", "Category"])
        writer.writerows(papers)

    print(f"Data extracted and saved in {output_path}")
