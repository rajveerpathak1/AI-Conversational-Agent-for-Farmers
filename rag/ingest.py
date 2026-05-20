import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

docs = []

folder_path = "data/agriculture_docs"

for file in os.listdir(folder_path):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(folder_path, file)

        loader = PyPDFLoader(pdf_path)

        loaded_docs = loader.load()

        for doc in loaded_docs:

            doc.metadata["file_name"] = file

            doc.metadata["page"] = doc.metadata.get("page", 0)

        docs.extend(loaded_docs)

print(f"Loaded {len(docs)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120
)

split_docs = splitter.split_documents(docs)

print(f"Created {len(split_docs)} chunks")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    split_docs,
    embeddings
)

vectorstore.save_local("vectorstore")

print("Vector DB created successfully")