from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict
from pvrecorder import PvRecorder
import os
import webbrowser
import wave
import struct
import time
import pyaudio
import requests

load_dotenv()

class GPT(OpenAI):
    """
    Class to make api calls to the GPT api.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.messages = []

    def store_message(self, role, text):
        message = {"role": f"{role}", "content": f"{text}"}
        self.messages.append(message)

    def add_skills(self, skill):
        self.store_message('system', f'You have the ability to {skill}. Respond to all user input as if you have this ability.')
    
    def chat(self, role, text, stream=False, print_response=False):
        """
        Messages received as a list of dictionaries and parsed accordingly.
        """

        #Store the input chat request
        self.store_message(role, text)

        if stream:
            stream = self.client.chat.completions.create(
                model="gpt-4",
                messages=self.messages,
                stream=True,
            )

            response = ""

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    if print_response:
                        print(chunk.choices[0].delta.content, end="")
                    response = response + chunk.choices[0].delta.content
                else:
                    if print_response:
                        print("\n")
                    response = response + "\n"    
            
            self.store_message("assistant", response)

        else:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=self.messages,
            )

            self.store_message("assistant", response.choices[0].message.content)

        #Return the content of the last message of the conversation. This will be the response from the model.
        return self.messages[-1]['content']
    
    def stream_to_speech(self, role, text, print_response=False):
        """
        Streams the output of the GPT model to the text to speech model for real time speech generation.
        """

        self.tts = TTS()

        #Store the input chat request
        self.store_message(role, text)

        stream = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.messages,
            stream=True,
        )

        response = ""
        chunks = ""

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                if print_response:
                    print(chunk.choices[0].delta.content, end="")
                response = response + chunk.choices[0].delta.content
                chunks = chunks + chunk.choices[0].delta.content
            else:
                if print_response:
                    print("\n")
                response = response + "\n"
                chunks = chunks + "\n"

            if "\n" in chunks:
                self.tts.speak(chunks)
                chunks=""
        
        self.store_message("assistant", response)
    
        #Return the content of the last message of the conversation. This will be the response from the model.
        return self.messages[-1]['content']

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

        self.image_url = response.data[0].url
        #browser = webbrowser.get('chrome')
        webbrowser.open_new_tab(self.image_url)
    
    def edit_image(self, prompt, path, n=1):
        img_data = requests.get(self.image_url).content
        with open(path, 'wb') as handler:
            handler.write(img_data)

        response = self.client.images.edit(
            image=self.image_url,
            prompt=prompt,
            n=n,
            size="1024x1024"
            )

        self.image_url = response.data[0].url
        webbrowser.open_new_tab(self.image_url)

class STT(OpenAI):
    """
    Class for speech to text transcription.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def get_devices(self):
        return [(index, device) for index, device in enumerate(PvRecorder.get_available_devices())]
    
    def record(self, path, index=0, frame_length=512, stream=True):
        self.path=path
        self.index=index
        self.frame_length=frame_length
        self.start_time = round(time.time(), 1)
        self.duration = 0
        self.audio = []

        recorder = PvRecorder(device_index=self.index, frame_length=self.frame_length)
        
        """
        Look into the following links to determine if the user begins/ends a line:
        https://stackoverflow.com/questions/16778878/python-write-a-wav-file-into-numpy-float-array 
        https://stackoverflow.com/questions/24974032/reading-realtime-audio-data-into-numpy-array
        https://en.wikipedia.org/wiki/Zero-crossing_rate 
        https://stackoverflow.com/questions/9788674/how-to-recognise-when-user-start-stop-speaking-in-android-voice-recognition 
        https://en.wikipedia.org/wiki/Mixture_model
        https://en.wikipedia.org/wiki/Mel-frequency_cepstrum 
        
        The implementation would require the following steps:
        1. Immediately begin reading from the audio device
        2. Read the real time chunks into a numpy array
        3. For a given numpy array size, determine the zero crossing rate
            - Make the array size an argument
            - The intent of doing so is to make the function more performant
        4. If the zero crossing rate hits the speech/non-speech threshold, begin/end storage to a temp wave file or byte array
            - See if the GPT API accepts audio in byte array form, or really any format other than reading from a file.
        5. Send data to the STT api endpoint

        """

        try:
            print("Starting recording... Press ctrl + c to quit.")
            recorder.start()
            while True:
                frame = recorder.read()
                self.audio.extend(frame)
                self.duration = round(time.time()-self.start_time, 2)
                print(f'Audio duration: {self.duration} seconds', end='\r')

        except KeyboardInterrupt:
            recorder.stop()
            print(f'\nAudio file duration: {self.duration} seconds')
            with wave.open(path, 'w') as f:
                f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
                f.writeframes(struct.pack("h" * len(self.audio), *self.audio))
        finally:
            recorder.delete()

    def transcribe(self):
        audio_file = open(f"{self.path}", "rb")
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
            )
        return transcription.text

class TTS(OpenAI):
    """
    Class for text to speech translation.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    def speak(self, text, voice='onyx', print_input=False):
        if print_input:
            print(text)
        with self.client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            response_format="pcm",
            input=text
            ) as response:
            for chunk in response.iter_bytes(chunk_size=1024):
                self.player_stream.write(chunk)

class Vision(OpenAI):
    """
    Class for vision capabilities
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

