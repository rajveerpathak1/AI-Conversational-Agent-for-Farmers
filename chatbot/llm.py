from langchain_groq import ChatGroq
from config.settings import Config

llm = ChatGroq(
    groq_api_key=Config.GROQ_API_KEY,
    model_name=Config.MODEL_NAME
)