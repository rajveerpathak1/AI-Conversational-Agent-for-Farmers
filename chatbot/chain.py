import os

from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

load_dotenv()

from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent AI agriculture assistant for Haryana farmers.

Answer ONLY using the provided context.

Rules:
- Keep answers practical and simple.
- Use bullet points when useful.
- Mention schemes clearly.
- If information is missing, say so honestly.
- Do not hallucinate.
- Explain agricultural terms simply.

Context:
{context}

Question:
{question}

Answer:
"""
)



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20
    }
)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={
        "prompt": custom_prompt
    },
    return_source_documents=True
)

def ask_question(query):

    response = qa_chain.invoke({
        "question": query
    })

    answer = response["answer"]

    sources = []

    for doc in response["source_documents"]:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")

        sources.append(f"{source} (Page {page})")

    return {
        "answer": answer,
        "sources": list(set(sources))
    }