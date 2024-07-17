import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate

load_dotenv()
key = os.getenv('openai-key')

# Create an instance of the Ollama model
llm = Ollama(model="llama-3")

# Define a function to use the model for text generation
def generate_text(prompt: str):
    response = llm.generate([prompt])
    return response[0]['text']

# Example usage
prompt = "Tell me a story about a brave knight."
response = generate_text(prompt)
print(response)
