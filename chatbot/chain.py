import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain.chains import ConversationalRetrievalChain


# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# CONFIG
# =========================================================

VECTORSTORE_PATH = "vectorstore"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

LLM_MODEL = "llama-3.3-70b-versatile"

TEMPERATURE = 0.2

TOP_K = 5
FETCH_K = 20

USE_GPU = False


# =========================================================
# CUSTOM PROMPT
# =========================================================

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert AI agriculture assistant helping farmers in Haryana, India.

Your job is to answer ONLY using the provided context.

Guidelines:
- Give practical and actionable answers first.
- Use simple conversational language.
- Be educational but concise.
- Organize answers using headings and bullet points.
- Mention risks, precautions, and limitations when relevant.
- Prefer Haryana-specific farming advice when available.
- If the context is incomplete, clearly say:
  "I could not find enough information in the provided documents."

Do NOT hallucinate.
Do NOT invent government schemes, prices, or recommendations.

Context:
{context}

Question:
{question}

Answer:
"""
)


# =========================================================
# EMBEDDINGS
# =========================================================

device = "cuda" if USE_GPU else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": device
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# =========================================================
# LOAD VECTOR DATABASE
# =========================================================

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)


# =========================================================
# RETRIEVER
# =========================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": TOP_K,
        "fetch_k": FETCH_K
    }
)


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=TEMPERATURE
)


# =========================================================
# MEMORY
# =========================================================

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    k=4
)

# Keeps only recent conversation
# Prevents token explosion


# =========================================================
# QA CHAIN
# =========================================================

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,

    retriever=retriever,

    memory=memory,

    combine_docs_chain_kwargs={
        "prompt": custom_prompt
    },

    return_source_documents=True,

    verbose=False
)


# =========================================================
# ASK QUESTION FUNCTION
# =========================================================

def ask_question(query: str):

    response = qa_chain.invoke({
        "question": query
    })

    answer = response["answer"]

    # =====================================================
    # FORMAT SOURCES
    # =====================================================

    sources = []

    seen = set()

    for doc in response["source_documents"]:

        file_name = doc.metadata.get(
            "file_name",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            0
        )

        citation = f"{file_name} (Page {page})"

        if citation not in seen:

            seen.add(citation)

            sources.append(citation)

    return {
        "answer": answer.strip(),
        "sources": sources
    }


# =========================================================
# CLI TEST LOOP
# =========================================================

if __name__ == "__main__":

    print("\n🌾 Haryana Agriculture AI Assistant")
    print("Type 'exit' to quit\n")

    while True:

        query = input("You: ")

        if query.lower() == "exit":
            break

        result = ask_question(query)

        print("\nAssistant:\n")
        print(result["answer"])

        print("\nSources:")
        for src in result["sources"]:
            print(f"- {src}")

        print("\n" + "=" * 60 + "\n")
