import sys
sys.path.append('..')
import src.openai_api as ai
import os

basepath = os.path.join(os.path.dirname(__file__), "audio")
filepath = os.path.join(basepath, f'audio_{len(os.listdir(basepath))}.wav')

#Instantiate the GPT, STT, and TTS models
gpt = ai.GPT()
stt = ai.STT()
tts = ai.TTS()

#Set up audio capture
devices = stt.get_devices()
index = devices[-1][0]

while True:
    #Record the user input
    stt.record(filepath, index)
    print("Transcribing...")
    #Transcribe the user input
    transcription = stt.transcribe()
    print("Transcription complete.")
    print("Generating response...")
    #Format the message then feed the user input to the GPT model
    response = gpt.stream_to_speech("user", transcription)

