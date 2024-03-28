from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("API_Key"))


class GPT():
    """
    Class to make api calls to the GPT api.
    """
    def __init__():
