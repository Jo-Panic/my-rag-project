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
