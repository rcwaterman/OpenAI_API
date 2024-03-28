from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
import os
import webbrowser
import urllib

load_dotenv()

class GPT(OpenAI):
    """
    Class to make api calls to the GPT api.
    """
    def __init__(self):
        #super().__init__(api_key=os.environ.get("OPENAI_API_KEY"))
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def chat(self, messages: List[Dict[str, str]]):
        """
        Messages received as a list of dictionaries and parsed accordingly.
        """
        stream = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="")
        print("\n")

class Dalle(OpenAI):
    """
    Class to make api calls to the GPT api.
    """
    def __init__(self):
        #super().__init__(api_key=os.environ.get("OPENAI_API_KEY"))
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def generate_image(self, prompt, size="1024x1024", quality="standard", n=1):
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
            )

        image_url = response.data[0].url
        webbrowser.open_new_tab(image_url)