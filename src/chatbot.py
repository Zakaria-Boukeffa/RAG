import argparse  
from .rag import RAGSystem
from .utils import get_llm_client

def main():
    rag = RAGSystem()
    print("Système RAG initialisé. Tapez 'exit' pour quitter.\n")
    
    while True:
        try:
            query = input("\nVous: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            print("\nAssistant: ", end="", flush=True)
            response = rag.generate(query)
            print(response)
            
        except KeyboardInterrupt:
            print("\nFin de la session.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot RAG LangChain")
    parser.add_argument('--test', action='store_true', help="Exécuter le test initial")
    args = parser.parse_args()
    
    if args.test:
        from .rag import RAGSystem # <-- Import corrigé
        rag = RAGSystem()
        print(rag.generate("Comment créer un chain avec LangChain ?"))
    else:
        main()
