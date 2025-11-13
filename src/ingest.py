import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
PGVECTOR_CONNECTION = os.getenv("PGVECTOR_CONNECTION")  
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_docs")

def ingest_pdf():
    if not PDF_PATH:
        raise ValueError("PDF_PATH não definido no .env")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        connection_string=PGVECTOR_CONNECTION,
        collection_name=COLLECTION_NAME,
        pre_delete_collection=True,  
    )

    print(f"Ingestão concluída! {len(docs)} chunks inseridos em '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    ingest_pdf()
