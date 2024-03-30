import sys
sys.path.append('..')
import src.openai_api

system = input("Enter the system context, then press enter: ")
user = input("Enter the user prompt, then press enter: ")

messages = [{"role": "system", "content": f"{system}"},
            {"role": "user", "content": f"{user}"}]

gpt_client = src.openai_api.GPT()

response = gpt_client.chat(messages)

dalle_client = src.openai_api.Dalle()

image = dalle_client.generate_image(prompt=response)
