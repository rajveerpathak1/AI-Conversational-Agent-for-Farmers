import os

from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain.prompts import PromptTemplate

load_dotenv()

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent AI agriculture assistant for Haryana farmers.

Answer ONLY using the provided context.

Rules:
- First provide a concise practical answer.
- Then provide detailed explanation if needed.
- Keep answers conversational and farmer-friendly.
- Mention when information is uncertain or incomplete.
- Prefer region-specific and farmer-specific recommendations.
- Prioritize practical schemes over generic explanations.
- Focus on actionable farmer benefits.
- Organize responses with headings and subheadings.
- Explain concepts in a structured educational way.
- Use examples where possible.
- Summarize retrieved information intelligently.
- Avoid simply listing raw extracted points.
- If asked for details, provide:
  • definition
  • uses
  • benefits
  • drawbacks
  • environmental impact
  • safety precautions

Context:
{context}

Question:
{question}

Answer:
"""
)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Vector DB
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20
    }
)

# LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# QA Chain
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

        citation = f"{file_name} (Page {page + 1})"

        if citation not in seen:

            seen.add(citation)

            sources.append(citation)

    return {
        "answer": answer,
        "sources": sources
    }