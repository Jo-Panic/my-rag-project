"""
Module de configuration du système RAG.
Contient l'initialisation des modèles et les paramètres globaux.

RAG system configuration module.
Contains model initialization and global parameters.
"""

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize embedding model using HuggingFace's all-mpnet-base-v2
EMBEDDING_MODEL = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Initialize local LLM using Ollama
# timeout set to 120s due to the large model size (70B parameters)
# You can change the model
# LLM = Ollama(model="llama3.3:70b-instruct-q6_K", request_timeout=120.0)
# LLM = Ollama(model="llama3.3:70b-instruct-q5_0", request_timeout=120.0)
LLM = Ollama(model="llama3.3:70b-instruct-q3_K_M", request_timeout=120.0)

# Global RAG settings
CHUNK_SIZE = 2048  # Large chunks to maintain context and section coherence
CHUNK_OVERLAP = 128  # Overlap between chunks to avoid losing context at boundaries
TOP_K = 8  # Number of relevant chunks to use


def init_settings():
    """Initialize global LlamaIndex settings"""
    Settings.llm = LLM
    Settings.embed_model = EMBEDDING_MODEL
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
