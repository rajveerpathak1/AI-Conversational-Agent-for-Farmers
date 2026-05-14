import os

from dotenv import load_dotenv

from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

def ask_question(query):

    response = qa_chain.invoke({
        "question": query
    })

    return response["answer"]