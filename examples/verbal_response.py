import sys
sys.path.append('..')
import src.openai_api as ai
import os

basepath = os.path.join(os.path.dirname(__file__), "audio")
filepath = os.path.join(basepath, f'audio_{len(os.listdir(basepath))}.wav')

#Ask for user input to create the system context
system = input("Enter the system context, then press enter: ")
user = input("Enter the user prompt, then press enter: ")

#Format the user input for the API
messages = [{"role": "system", "content": f"{system}"},
            {"role": "user", "content": f"{user}"}]

#Instantiate the GPT and TTS models
gpt = ai.GPT()
tts = ai.TTS()

#Provide the model context
response = gpt.chat(messages)

#feed the response into the TTS model
tts.speak(response)
