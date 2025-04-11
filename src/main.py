import argparse
import sys
import os
import gradio as gr  # <-- Import Gradio

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag import RAGSystem  # <-- Import direct
from utils import get_llm_client

def create_interface():
    rag = RAGSystem()

    def respond(message, chat_history):
        try:
            bot_message = rag.generate(message)
        except Exception as e:
            bot_message = f"Erreur : {e}"
        chat_history.append((message, bot_message))
        return "", chat_history

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(show_label=False, placeholder="Tapez votre question ici...", container=False)
        clear = gr.Button("Effacer")

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    return demo

def main():
    parser = argparse.ArgumentParser(description="Chatbot RAG LangChain")
    parser.add_argument('--gradio', action='store_true', help="Lancer l'interface Gradio")
    args = parser.parse_args()

    if args.gradio:
        interface = create_interface()
        interface.launch()
    else:
        rag = RAGSystem()  # <-- Initialisation en mode CLI
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
    main()
