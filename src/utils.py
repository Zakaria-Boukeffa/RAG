from openai import OpenAI

DEEPINFRA_API_TOKEN = "#"
DEEPINFRA_MODEL_ID = "#"

def get_llm_client():
    return OpenAI(
        api_key=DEEPINFRA_API_TOKEN,
        base_url="https://api.deepinfra.com/v1/openai",
    )

def chat_with_llm(client):
    """Permet une interaction interactive avec le LLM."""
    while True:
        user_input = input("Vous : ")
        if user_input.lower() == "exit":
            break
        try:
            response = client.chat.completions.create(
                model=DEEPINFRA_MODEL_ID,
                messages=[{"role": "user", "content": user_input}],
            )
            print("LLM :", response.choices[0].message.content)
        except Exception as e:
            print(f"Erreur : {e}")

if __name__ == "__main__":
    print("Démarrage de la conversation avec le LLM (tapez 'exit' pour quitter).")
    client = get_llm_client()
    chat_with_llm(client)
    print("Conversation terminée.")
    print("Démarrage de la conversation avec le LLM (tapez 'exit' pour quitter).")
    client = get_llm_client()
    chat_with_llm(client)
    print("Conversation terminée.")