import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path("data/agriculture_docs")
VECTORSTORE_DIR = "vectorstore"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

USE_GPU = False  # Set True if CUDA is available


# =========================================================
# LOAD PDF DOCUMENTS
# =========================================================

print("\nLoading PDF documents...\n")

loader = DirectoryLoader(
    path=str(DATA_DIR),
    glob="*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True
)

try:
    docs = loader.load()
except Exception as e:
    raise RuntimeError(f"Error loading documents: {e}")

print(f"\nLoaded {len(docs)} pages")


# =========================================================
# CLEAN + ENRICH METADATA
# =========================================================

for idx, doc in enumerate(docs):

    source_path = doc.metadata.get("source", "")

    file_name = os.path.basename(source_path)

    doc.metadata.update({
        "file_name": file_name,
        "source": source_path,
        "page": doc.metadata.get("page", 0) + 1,
        "document_type": "pdf",
        "doc_id": idx
    })


# =========================================================
# SPLIT DOCUMENTS
# =========================================================

print("\nSplitting documents into chunks...\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)

split_docs = splitter.split_documents(docs)

# Add chunk ids
for idx, doc in enumerate(split_docs):
    doc.metadata["chunk_id"] = idx

print(f"Created {len(split_docs)} chunks")


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...\n")

device = "cuda" if USE_GPU else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

print(f"Using device: {device}")
print(f"Embedding model: {EMBEDDING_MODEL}")


# =========================================================
# CREATE VECTOR STORE
# =========================================================

print("\nCreating FAISS vector database...\n")

vectorstore = FAISS.from_documents(
    documents=split_docs,
    embedding=embeddings
)


# =========================================================
# SAVE VECTOR STORE
# =========================================================

os.makedirs(VECTORSTORE_DIR, exist_ok=True)

vectorstore.save_local(VECTORSTORE_DIR)

print(f"\nVector database saved successfully at '{VECTORSTORE_DIR}'")


# =========================================================
# OPTIONAL TEST QUERY
# =========================================================

print("\nRunning test similarity search...\n")

query = "best fertilizer for wheat crop"

results = vectorstore.similarity_search(query, k=3)

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("-" * 50)

    print(f"File : {result.metadata.get('file_name')}")
    print(f"Page : {result.metadata.get('page')}")

    print("\nContent Preview:")
    print(result.page_content[:300])
