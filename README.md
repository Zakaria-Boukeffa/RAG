# Chatbot-RAG

## Description

Ce projet implémente un chatbot avancé utilisant l'architecture RAG (Retrieval Augmented Generation) pour répondre à des questions sur un corpus de données techniques liées à l'intelligence artificielle, en particulier la documentation de LangChain.

## Architecture

L'architecture du chatbot est divisée en plusieurs composants :

*   **Chargement des données :**  Charge et prépare les documents à partir du corpus.
*   **Chunking :**  Divise les documents en segments de texte (chunks) de taille appropriée.
*   **Embedding :**  Génère des embeddings vectoriels pour chaque chunk à l'aide du modèle `sentence-transformers/all-MiniLM-L6-v2`.
*   **Base de données vectorielle :**  Stocke les embeddings dans une base de données vectorielle Chroma pour une recherche rapide.
*   **RAG :**  Récupère les chunks les plus pertinents en fonction de la question de l'utilisateur et les utilise pour générer une réponse à l'aide d'un LLM.

## Choix techniques

*   **Langage :**  Python 3.8+
*   **Modèle d'embedding :**  `sentence-transformers/all-MiniLM-L6-v2`
*   **Base de données vectorielle :**  Chroma
*   **Framework :**  LangChain

## Installation

1.  Clonez le repository :

    ```
    git clone https://github.com/Zakaria-Boukeffa/RAG.git
    cd rag-chatbot
    ```
2.  Créez un environnement virtuel (recommandé) :

    ```
    python -m venv venv
    source venv/bin/activate  # ou venv\Scripts\activate sur Windows
    ```
3.  Installez les dépendances :

    ```
    pip install -r requirements.txt
    ```
4.  Configurez votre clé API DeepInfra dans le fichier `src/utils.py`.

## Utilisation

1.  **Exécutez le script d'embedding pour créer la base de données vectorielle :**

    ```
    python src/main.py embedding
    ```
2.  **Lancez le chatbot en mode CLI :**

    ```
    python src/main.py chatbot
    ```
    **Ou lancez le chatbot avec une interface Gradio:**

    ```
    python src/main.py chatbot --gradio
    ```
    Et apres aller en url local

## Structure du code
```bash
rag-chatbot/
├── README.md
├── requirements.txt
├── RAPPORT.md
├── data/
│ └── train.jsonl
├── src/
│ ├── main.py
│ ├── embedding.py
│ ├── rag.py
│ ├── chatbot.py
│ └── utils.py
└── tests/
└── test_queries.py
```

## Démonstration fonctionnelle

Une fois l'installation terminée, vous pouvez interagir avec le chatbot en utilisant l'interface en ligne de commande. Tapez votre question et le chatbot vous fournira une réponse basée sur le corpus de données.
