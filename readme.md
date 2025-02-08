# (Very Little) RAG Markdown Assistant

[English](#english) | [Français](#français)

## Français

### 📝 Description

Un système RAG (Retrieval Augmented Generation) local pour interroger vos notes Markdown. Utilise Ollama pour le LLM, ChromaDB pour le stockage des embeddings, et LlamaIndex pour l'orchestration.

### 🛠 Prérequis

- Python 3.10+
- Conda
- [Ollama](https://ollama.ai) installé et configuré
- Un modèle LLM compatible installé via Ollama (ex: llama2)

### 🚀 Installation

1. Cloner le dépôt :

```bash
git clone https://github.com/Jo-Panic/my-rag-project
cd my-rag-project
```

2. Créer l'environnement conda :

```bash
conda create -n rag-env python=3.10
conda activate rag-env
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

### 💻 Utilisation

1. Placez vos fichiers Markdown dans le dossier `docs/`
2. Lancez le script :

```bash
python src/main.py
```

### 📁 Structure du Projet

```
.
├── docs/                    # Vos fichiers Markdown
├── src/                    # Code source
│   ├── main.py            # Script principal
│   ├── config.py          # Configuration et initialisation
│   ├── markdown_processor.py   # Traitement des fichiers Markdown
│   ├── prompts.py         # Prompts système
│   └── query_engine.py    # Moteur de requête et validation
├── chroma_db/             # Base de données vectorielle (généré)
├── requirements.txt       # Dépendances Python
└── README.md             # Ce fichier
```

---

## English

### 📝 Description

A local RAG (Retrieval Augmented Generation) system to query your Markdown notes. Uses Ollama for LLM, ChromaDB for embeddings storage, and LlamaIndex for orchestration.

### 🛠 Prerequisites

- Python 3.10+
- Conda
- [Ollama](https://ollama.ai) installed and configured
- A compatible LLM model installed via Ollama (e.g., llama2)

### 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/Jo-Panic/my-rag-project
cd my-rag-project
```

2. Create conda environment:

```bash
conda create -n rag-env python=3.10
conda activate rag-env
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 💻 Usage

1. Place your Markdown files in the `docs/` folder
2. Run the script:

```bash
python src/main.py
```

### 📁 Project Structure

```
.
├── docs/                    # Your Markdown files
├── src/                    # Source code
│   ├── main.py            # Main script
│   ├── config.py          # Configuration and initialization
│   ├── markdown_processor.py   # Markdown file processing
│   ├── prompts.py         # System prompts
│   └── query_engine.py    # Query engine and validation
├── chroma_db/             # Vector database (generated)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```
