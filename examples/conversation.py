import sys
sys.path.append('..')
import src.openai_api as ai
import os

path = os.path.join(os.path.dirname(__file__), "audio")

stt = ai.STT()
devices = stt.get_devices()
print(devices)
index = devices[-1][0]
stt.record(os.path.join(path, f'audio_{len(os.listdir(path))}.wav'), index)
exit()

#Ask for user input to create the system context
system = input("Enter the system context, then press enter: ")

#Format the user input into a message block
messages = [{"role": "system", "content": f"{system}"}]

#Instantiate the GPT, STT, and TTS models
gpt = ai.GPT()
stt = ai.STT()
tts = ai.TTS()

#Provide the model context
gpt.chat(messages)
