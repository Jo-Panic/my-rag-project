"""
Module pour le moteur de requête.
Contient la logique de validation et de génération des réponses.

Query engine module.
Contains the validation logic and response generation.
"""

from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from prompts import get_validation_prompt, get_response_prompt
from config import TOP_K


def create_query_engine_with_validation(index: VectorStoreIndex, llm: Ollama):
    """
    Creates an enhanced query engine that validates the relevance of retrieved passages
    before generating a response. This helps ensure that responses are only generated
    when relevant information is found in the documents.

    Args:
        index (VectorStoreIndex): The vector store index containing the document embeddings
        llm (Ollama): The LLM instance used for both validation and response generation

    Returns:
        Callable: A query function that includes relevance validation
    """

    def check_relevance(question: str, nodes) -> bool:
        """
        Validates if the retrieved passages contain relevant information for the question.
        Uses the LLM to perform a binary classification task.

        Args:
            question (str): The user's question
            nodes: Retrieved document nodes from the vector store

        Returns:
            bool: True if the passages are deemed relevant, False otherwise
        """
        context = "\n\n".join(
            [f"Passage {i + 1}:\n{node.text}" for i, node in enumerate(nodes)]
        )

        # The classification prompt is designed to:
        # 1. Be strict about relevance
        # 2. Handle different types of questions (how-to, definitions, procedures)
        # 3. Consider information spread across multiple passages
        # Get the validation prompt from prompts.py
        prompt = get_validation_prompt(question, context)
        # Get classification from LLM and check for positive response
        response = llm.complete(prompt).text.strip().upper()
        return "OUI" in response

    # Create base query engine with specific parameters
    base_query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        response_mode="tree_summarize",
        system_prompt=get_response_prompt(),
    )

    def query_with_validation(question: str):
        """
        Enhanced query function that includes relevance validation before generating responses.

        Args:
            question (str): The user's question

        Returns:
            str: Either the answer or a message indicating no relevant information found
        """
        # First retrieve relevant nodes
        retriever = index.as_retriever(similarity_top_k=8)
        nodes = retriever.retrieve(question)

        # Check if retrieved passages are relevant
        if check_relevance(question, nodes):
            # If relevant, use base query engine to generate response
            return base_query_engine.query(question)
        else:
            # If not relevant, return standard "no information" message
            return "Je ne trouve pas dans les documents les informations nécessaires pour répondre à cette question."

    return query_with_validation
