from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("API_Key"))

class GPT(OpenAI):
    """
    Class to make api calls to the GPT api.
    """
    def __init__(self):
        self.client = super().__init__()

    def chat(self, messages):
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="")