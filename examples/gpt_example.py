from src.openai_api import GPT
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

message = [{"role": "user", "content": "Say this is a test"}]

client = GPT(OpenAI(api_key=os.getenv("API_Key")))

client.chat(message)