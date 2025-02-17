"""
Module de configuration du système RAG.
Contient l'initialisation des modèles et les paramètres globaux.

RAG system configuration module.
Contains model initialization and global parameters.
"""

from pathlib import Path
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

# Initialize embedding model using HuggingFace's all-mpnet-base-v2
EMBEDDING_MODEL = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Initialize local LLM using Ollama
# timeout set to 120s due to the large model size (70B parameters)
# You can change the model
# LLM = Ollama(model="llama3.3:70b-instruct-q6_K", request_timeout=120.0)
# LLM = Ollama(model="llama3.3:70b-instruct-q5_0", request_timeout=120.0)
# LLM = Ollama(model="llama3.3:70b-instruct-q3_K_M", request_timeout=120.0)
LLM = Ollama(model="llama3.3:70b-instruct-q2_K", request_timeout=120.0)

# Global RAG settings
CHUNK_SIZE = 2048  # Large chunks to maintain context and section coherence
CHUNK_OVERLAP = 128  # Overlap between chunks to avoid losing context at boundaries
TOP_K = 8  # Number of relevant chunks to use

# ChromaDB settings
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "markdown_docs"


def init_chroma_storage():
    """Initialize ChromaDB storage"""
    # Ensure the persist directory exists
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    # Create ChromaDB client with persistence
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Get or create collection
    chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return storage_context


def init_settings():
    """Initialize global LlamaIndex settings"""
    Settings.llm = LLM
    Settings.embed_model = EMBEDDING_MODEL
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
