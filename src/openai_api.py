from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
from pvrecorder import PvRecorder
import os
import webbrowser
import urllib
import wave
import struct
import time

load_dotenv()

class GPT(OpenAI):
    """
    Class to make api calls to the GPT api.
    """
    def __init__(self):
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

        response = ""

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                response = response + chunk.choices[0].delta.content
            else:
                print("\n")
                response = response + "\n"       
        return response

class Dalle(OpenAI):
    """
    Class to make api calls to the GPT api.
    """
    def __init__(self):
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
        #browser = webbrowser.get('chrome')
        webbrowser.open_new_tab(image_url)

class STT(OpenAI):
    """
    Class for speech to text transcription.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def get_devices(self):
        return [(index, device) for index, device in enumerate(PvRecorder.get_available_devices())]
    
    def record(self, path, index=0, frame_length=512):
        self.path = path
        self.index=index
        self.frame_length=frame_length
        self.start_time = round(time.time(), 3)
        self.duration = 0
        self.audio = []

        recorder = PvRecorder(device_index=self.index, frame_length=self.frame_length)
        
        try:
            print("Starting recording...")
            recorder.start()
            while True:
                frame = recorder.read()
                self.audio.extend(frame)
                if round(self.start_time - round(time.time(), 3), 1) == 0.1:
                    self.duration += self.duration+0.1
                    print(f'Audio duration: {self.duration} seconds')

        except KeyboardInterrupt:
            recorder.stop()
            print(f'\nAudio file duration: {self.duration} seconds')
            with wave.open(path, 'w') as f:
                f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
                f.writeframes(struct.pack("h" * len(self.audio), *self.audio))
        finally:
            recorder.delete()

class TTS(OpenAI):
    """
    Class for text to speech translation.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
