"""
Point d'entrée principal du système RAG.
Gère l'interaction avec l'utilisateur et orchestre les différents composants.

Main entry point of the RAG system.
Handles user interaction and orchestrates different components.
"""

from llama_index.core import VectorStoreIndex
from config import init_settings, LLM
from markdown_processor import load_markdown_files
from query_engine import create_query_engine_with_validation


def main():
    # Initialize global settings
    init_settings()

    # Load and process all markdown files
    documents = load_markdown_files("docs")

    # Create vector index from documents
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine with validation
    query_engine = create_query_engine_with_validation(index, LLM)

    # Main interaction loop
    print("\nAssistant RAG initialisé. Posez vos questions sur les documents !")

    while True:
        question = input("\nVotre question (ou 'q' pour quitter) : ")
        if question.lower() == "q":
            break

        # Get and display the response
        response = query_engine(question)
        print("\nRéponse :", response)


if __name__ == "__main__":
    main()
