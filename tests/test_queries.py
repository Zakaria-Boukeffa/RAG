# tests/test_queries.py
import unittest
import sys
import os

# Ajouter le dossier src au PYTHONPATH pour les imports relatifs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.rag import RAGSystem  # <-- Import corrigé
from time import time

class TestRAGSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = RAGSystem()
        cls.test_queries = [
            ("Comment créer un chain avec LangChain ?", ["LLMChain", "PromptTemplate"]),
            ("Qu'est-ce qu'un Memory module ?", ["ConversationBufferMemory", "EntityMemory"]),
            ("Comment connecter LangChain à une base SQL ?", ["SQLDatabaseChain", "SQLAlchemy"])
        ]
    
    def test_response_quality(self):
        for query, keywords in self.test_queries:
            with self.subTest(query=query):
                start_time = time()
                response = self.rag.generate(query)
                latency = time() - start_time
                
                print(f"\nTest: {query}")
                print(f"Latence: {latency:.2f}s")
                print(f"Réponse:\n{response}")
                
                # Vérification des mots-clés
                missing_keywords = [kw for kw in keywords if kw.lower() not in response.lower()]
                self.assertEqual(len(missing_keywords), 0, 
                               f"Mots-clés manquants: {missing_keywords}")

if __name__ == "__main__":
    unittest.main()
