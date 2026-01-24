import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def get_openai_key():
    # 1. Check if already in environment (e.g., passed via CLI)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # 2. Try to get from Colab Secrets (only works in interactive cells)
    try:
        from google.colab import userdata
        return userdata.get('OPENAI_API_KEY')
    except (ImportError, Exception):
        return None

def get_google_key():
    # 1. Check if already in environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key
    
    # 2. Try to get from Colab Secrets
    try:
        from google.colab import userdata
        return userdata.get('GOOGLE_API_KEY')
    except (ImportError, Exception):
        return None

# Setup OpenAI Key
openai_key = get_openai_key()
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

# Setup Google Key
google_key = get_google_key()
if google_key:
    os.environ["GOOGLE_API_KEY"] = google_key

# Centralized LLM Instance
# Default to Google Gemini, but keep OpenAI as an option

LLM_MODEL = "gpt-4o"
# Option 1: OpenAI
llm = ChatOpenAI(model=LLM_MODEL)

# Option 2: Google Gemini
#LLM_MODEL = "gemini-2.0-flash"
#llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=google_key)
