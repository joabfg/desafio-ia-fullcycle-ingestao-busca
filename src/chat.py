import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import PGVector
from search import PROMPT_TEMPLATE

load_dotenv()

PGVECTOR_CONNECTION = os.getenv("PGVECTOR_CONNECTION")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_docs")

def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PGVector(
        connection_string=PGVECTOR_CONNECTION,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

def montar_prompt(contexto, pergunta):
    return PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)

def main():
    vectorstore = get_vectorstore()
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    print("Digite sua pergunta (ou 'sair' para encerrar):")
    while True:
        pergunta = input("> ").strip()
        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("Encerrando chat.")
            break

        docs = vectorstore.similarity_search(pergunta, k=10)
        contexto = "\n\n".join([doc.page_content for doc in docs])

        prompt = montar_prompt(contexto, pergunta)
        resposta = llm.invoke(prompt)
        print(f"\nResposta:\n{resposta.content}\n")

if __name__ == "__main__":
    main()