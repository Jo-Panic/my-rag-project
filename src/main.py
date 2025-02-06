from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path

# Initialize embedding model using HuggingFace's all-mpnet-base-v2
# This model provides better semantic understanding compared to simpler models
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize local LLM using Ollama
# timeout set to 120s due to the large model size (70B parameters)
# You can change the model
llm = Ollama(model="llama3.3:70b-instruct-q6_K", request_timeout=120.0)

# Global settings for the RAG system
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 2048  # Large chunks to maintain context and section coherence
Settings.chunk_overlap = (
    128  # Overlap between chunks to avoid losing context at boundaries
)


def split_markdown_by_headers(content: str) -> list[str]:
    """
    Split markdown content into sections based on h2 headers (##)
    This preserves the document structure and maintains context within sections

    Args:
        content: Raw markdown content
    Returns:
        List of sections, each starting with an h2 header
    """
    sections = []
    current_section = []

    for line in content.split("\n"):
        if line.startswith("## "):  # New section starts with h2 header
            if current_section:
                sections.append("\n".join(current_section))
            current_section = [line]
        else:
            current_section.append(line)

    # Don't forget the last section
    if current_section:
        sections.append("\n".join(current_section))

    return sections


def load_markdown_files(directory: str) -> list[Document]:
    """
    Load and process markdown files from a directory
    Splits each file into sections and creates Document objects

    Args:
        directory: Path to the documents directory
    Returns:
        List of Document objects, each representing a section
    """
    documents = []
    for path in Path(directory).rglob("*.md"):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            # Split content into sections
            sections = split_markdown_by_headers(content)

            # Create a document for each section
            for section in sections:
                if section.strip():  # Skip empty sections
                    documents.append(
                        Document(
                            text=section,
                            metadata={
                                "source": str(path),
                                "section": section.split("\n")[0]
                                if section.split("\n")
                                else "Introduction",
                            },
                        )
                    )
    return documents


def main():
    # Load and process all markdown files
    documents = load_markdown_files("docs")

    # Create vector index from documents
    index = VectorStoreIndex.from_documents(documents)

    # Configure query engine with specific parameters
    query_engine = index.as_query_engine(
        similarity_top_k=8,  # Number of relevant chunks to use (balance between context and precision)
        response_mode="tree_summarize",  # Use hierarchical summarization for more coherent responses
        system_prompt="""Vous êtes un assistant expert qui doit répondre aux questions en utilisant UNIQUEMENT les informations des documents fournis.
    Si vous voyez une information pertinente dans les documents, même si elle n'est pas formulée exactement comme la question, utilisez-la.
    Cherchez particulièrement les sections qui contiennent des informations liées à la question, même si les termes exacts ne correspondent pas.""",
    )

    # Main interaction loop
    while True:
        question = input("\nVotre question (ou 'q' pour quitter) : ")
        if question.lower() == "q":
            break

        # DEBUG SECTION START
        # This section helps understand which documents are being used for answers
        # retriever = index.as_retriever(similarity_top_k=5)
        # nodes = retriever.retrieve(question)
        # print("\nSections pertinentes trouvées :")
        # for i, node in enumerate(nodes, 1):
        #     print(f"\nSection {i}:")
        #     print(f"Source: {node.metadata['source']}")
        #     print(f"Section: {node.metadata['section']}")
        #     print(f"Score: {node.score if hasattr(node, 'score') else 'N/A'}")
        #     print("Contenu:", node.text[:200], "...")
        # DEBUG SECTION END

        # Get and display the response
        response = query_engine.query(question)
        print("\nRéponse :", response)


if __name__ == "__main__":
    main()
