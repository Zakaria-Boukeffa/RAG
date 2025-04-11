import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from src.utils import get_llm_client

class RAGSystem:
    def __init__(self):
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_collection("langchain_docs")
        self.llm_client = get_llm_client()
        self.last_retrieved_ids = []
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert en LangChain. Réponds à la question en utilisant exclusivement le contexte suivant :
            
            {context}
            
            Réponse :"""),
            ("user", "{question}")
        ])
    
    def retrieve(self, query, k=5):
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"]  # <-- Pas de "ids" ici
        )
        
        # IDs disponibles même sans les demander
        self.last_retrieved_ids = results["ids"][0]  # <-- Clé "ids" toujours présente
        
        return "\n\n---\n\n".join(
            f"Document {i+1} (ID: {self.last_retrieved_ids[i]}):\n{text}"
            for i, text in enumerate(results["documents"][0])
        )

    
    def generate(self, question):
        context = self.retrieve(question)
        used_ids = ", ".join(self.last_retrieved_ids)
        
        messages = self.prompt.format_messages(
            context=context + f"\n\nIDs des documents utilisés: {used_ids}",
            question=question
        )
        
        response = self.llm_client.chat.completions.create(
            model="mistralai/Mistral-Small-24B-Instruct-2501",
            messages=[{"role": "system", "content": messages[0].content}]
            + [{"role": "user", "content": messages[1].content}],
            temperature=0.3,
            max_tokens=512,
        )
        
        return response.choices[0].message.content

if __name__ == "__main__":
    rag = RAGSystem()
    print("RAG system initialized")
    
    test_query = "Comment créer un chain avec LangChain ?"
    print(f"Testing query: {test_query}")
    print(rag.generate(test_query))
