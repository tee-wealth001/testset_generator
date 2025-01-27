from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI



# Load environment variables
load_dotenv()
gemini_api_token = os.getenv('GEMINI_API_KEY')
huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

TASK_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=gemini_api_token,
    temperature=0,
    format="json",
)