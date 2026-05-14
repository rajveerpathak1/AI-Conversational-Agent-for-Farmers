from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI agricultural assistant.

Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
)