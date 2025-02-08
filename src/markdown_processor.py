"""
Module pour le traitement des fichiers Markdown.
Contient les fonctions de chargement et de découpage des documents.

Markdown processing module.
Contains functions for loading and splitting documents.
"""

from pathlib import Path
from llama_index.core import Document


def split_markdown_by_headers(content: str, source_path: str) -> list[Document]:
    """
    Enhanced markdown content splitter that preserves hierarchy and handles code blocks.
    Only splits at level 2 headers (##) but preserves subheaders in content.

    Args:
        content: Raw markdown content
        source_path: Path to the source file
    Returns:
        List of Document objects
    """
    documents = []
    current_section = []
    current_title = "Introduction"  # Default title if section doesn't have title
    in_code_block = False

    for line in content.split("\n"):
        # Handle code blocks
        if line.startswith("```"):
            in_code_block = not in_code_block
            current_section.append(line)
            continue

        if in_code_block:
            current_section.append(line)
            continue

        # Only split on level 2 headers (##)
        if line.startswith("## "):
            # Save previous section if it exists
            if current_section:
                section_text = "\n".join(current_section)
                if section_text.strip():
                    documents.append(
                        Document(
                            text=section_text,
                            metadata={
                                "source": str(source_path),
                                "section": current_title,
                            },
                        )
                    )

            current_title = line[3:].strip()  # Remove "## " prefix
            current_section = [line]
        else:
            current_section.append(line)

    # Don't forget the last section
    if current_section:
        section_text = "\n".join(current_section)
        if section_text.strip():
            documents.append(
                Document(
                    text=section_text,
                    metadata={"source": str(source_path), "section": current_title},
                )
            )

    return documents


def load_markdown_files(directory: str) -> list[Document]:
    """
    Load and process markdown files from a directory

    Args:
        directory: Path to the documents directory
    Returns:
        List of Document objects
    """
    documents = []
    for path in Path(directory).rglob("*.md"):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.extend(split_markdown_by_headers(content, str(path)))
    return documents
