from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.llms import HuggingFaceHub



# Load environment variables
load_dotenv()
gemini_api_token = os.getenv('GEMINI_API_KEY')
huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')


LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=gemini_api_token,
    temperature=0.2
)

TASK_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=gemini_api_token,
    temperature=0,
    format="json",
)

EMBEDDINGS_LLM = GoogleGenerativeAIEmbeddings(
    # model="models/embedding-001",
    model="models/text-embedding-004",
    google_api_key=gemini_api_token
)

# Creating LLM using LangChain
HUG_LLM = HuggingFaceHub(
        repo_id="huggingfaceh4/zephyr-7b-alpha",
        huggingfacehub_api_token=huggingfacehub_api_token,
        model_kwargs={"temperature": 0.2, "max_length": 64, "max_new_tokens": 200}
    )