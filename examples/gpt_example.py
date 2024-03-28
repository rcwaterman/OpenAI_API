import sys
sys.path.append('..')
import src.openai_api

messages = [{"role": "system", "content": "You are a professor at MIT that teaches about machine learning, artificial intelligence, and large language models."},
            {"role": "user", "content": "What is the process of fine tuning a large language model like ChatGPT on user data?"}]

client = src.openai_api.GPT()

client.chat(messages)