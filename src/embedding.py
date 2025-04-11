import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import chromadb

def load_data(file_path="data/train.jsonl"):
    """Charge les données depuis un fichier JSONL."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get("text")  # Récupérer le texte
                source = data.get("source")  # Récupérer la source
                if text:
                    metadata = {"source": source} if source else {}
                    document = Document(page_content=text, metadata=metadata)
                    documents.append(document)
            except json.JSONDecodeError as e:
                print(f"Erreur de décodage JSON: {e}")
    return documents

def split_documents(docs):
    """Divise les documents en chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=102,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def create_vector_store(docs):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("langchain_docs")
    embeddings = model.encode([doc.page_content for doc in docs])
    collection.add(
        ids=[str(i) for i in range(len(docs))],
        embeddings=embeddings.tolist(),
        documents=[doc.page_content for doc in docs],
        metadatas=[doc.metadata for doc in docs]
    )
    return collection

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_data()
    print(f"Loaded {len(docs)} documents")
    print("\nSplitting documents...")
    splitted = split_documents(docs)
    print(f"Created {len(splitted)} chunks")
    print("\nCreating vector store...")
    collection = create_vector_store(splitted)
    print("Vector store created successfully")
    print(f"Collection count: {collection.count()}")
