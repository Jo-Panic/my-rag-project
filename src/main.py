"""
Point d'entrée principal du système RAG.
Gère l'interaction avec l'utilisateur et orchestre les différents composants.

Main entry point of the RAG system.
Handles user interaction and orchestrates different components.
"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.theme import Theme
from rich.prompt import Prompt

from llama_index.core import VectorStoreIndex
from config import init_settings, init_chroma_storage, LLM
from markdown_processor import load_markdown_files
from query_engine import create_query_engine_with_validation

# Console theme
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "success": "green",
        "user": "blue",
        "assistant": "green",
    }
)

# Console init with the theme
console = Console(theme=custom_theme)


def display_header():
    """Display a stylized header for the application"""
    header = Panel(
        """[success]Assistant RAG Local[/]
        
Prêt à répondre à vos questions sur les documents.""",
        title="💡 Bienvenue",
        border_style="blue",
    )
    console.print(header)
    console.print()


def display_startup_info(count):
    """Display startup information"""
    info = Panel(
        f"[info]{count}[/] embeddings chargés depuis ChromaDB",
        title="📊 État de l'index",
        border_style="cyan",
    )
    console.print(info)
    console.print()


def get_or_create_index(storage_context):
    """Get existing index or create a new one if needed"""
    # Get the ChromaDB collection from the storage context
    chroma_collection = storage_context.vector_store._collection

    # Check if the collection actually contains data
    try:
        count = chroma_collection.count()
        if count > 0:
            console.print("[success]Index existant détecté.[/]")
            index = VectorStoreIndex.from_vector_store(storage_context.vector_store)
            display_startup_info(count)
            return index
        else:
            console.print("[warning]Aucun document indexé trouvé.[/]")
            with console.status("[bold yellow]Création d'un nouvel index..."):
                documents = load_markdown_files("docs")
                index = VectorStoreIndex.from_documents(
                    documents, storage_context=storage_context
                )
            console.print("[success]Index créé avec succès ![/]")
            return index
    except Exception as e:
        console.print(f"[error]Erreur lors de la vérification de l'index : {e}[/]")
        with console.status("[bold yellow]Création d'un nouvel index..."):
            documents = load_markdown_files("docs")
            return VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )


def main():
    # Initialize global settings
    with console.status("[bold yellow]Initialisation du système..."):
        init_settings()
        # Initialize ChromaDB storage
        storage_context = init_chroma_storage()

    # Create vector index from documents with ChromaDB storage
    index = get_or_create_index(storage_context)

    # Create query engine with validation
    query_engine = create_query_engine_with_validation(index, LLM)

    # Display welcome header
    display_header()

    # Main interaction loop
    while True:
        # Get user input with styled prompt
        question = Prompt.ask(
            "\n[user]Votre question[/]", default="q", show_default=False
        )

        if question.lower() == "q":
            # Display goodbye message
            console.print("\n[info]Au revoir ! 👋[/]\n")
            break

        # Show thinking animation
        with Live(
            Spinner("dots", text="Recherche et analyse en cours..."),
            refresh_per_second=10,
        ):
            response = query_engine(question)

        # Display the response in a panel with markdown support
        response_panel = Panel(
            Markdown(str(response)), title="🤖 Réponse", border_style="green"
        )
        console.print("\n", response_panel)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[warning]Programme interrompu par l'utilisateur.[/]\n")
    except Exception as e:
        console.print(f"\n[error]Une erreur est survenue : {e}[/]\n")
