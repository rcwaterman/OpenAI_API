import sys
sys.path.append('..')
import src.openai_api as ai
import os
from src.utils.keywords import GENERATE_IMAGE_KEYWORDS, EDIT_IMAGE_KEYWORDS

basepath = os.path.join(os.path.dirname(__file__), "audio")
filepath = os.path.join(basepath, f'temp.wav')

#Instantiate the GPT, STT, and TTS models
gpt = ai.GPT()
stt = ai.STT()
dalle = ai.Dalle()

#Set up audio capture
devices = stt.get_devices()
index = devices[-1][0]

while True:
    #Record the user input
    stt.record(filepath, index)
    #Transcribe the user input
    transcript = stt.transcribe()

    print(transcript)

    phrases = []
    image=False
    chat=False
    edit=False

    #if a keyword is identified, add it to the text model's list of skills.
    for phrase in GENERATE_IMAGE_KEYWORDS:
        if phrase in transcript.lower():
            gpt.add_skills(phrase)
            phrases.append(phrase)
            image=True

    for phrase in phrases:    
        if phrase == 'describe':
            #Format the message then feed the user input to the GPT model
            response = gpt.stream_to_speech("user", transcript, print_response=True)
            dalle.generate_image(prompt=response)
            image=False
            chat=True

    for phrase in EDIT_IMAGE_KEYWORDS:
        if phrase in transcript.lower():
            gpt.add_skills(phrase)
            dalle.edit_image(transcript, r'./images/temp.png')
            image=False    

    if image == True:
        dalle.generate_image(prompt=transcript)
    if chat == False:
        response = gpt.stream_to_speech("user", transcript, print_response=True)


