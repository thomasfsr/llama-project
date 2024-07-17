from huggingface_hub import login
from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

load_dotenv()
key = os.getenv('key')

login(key)
app = FastAPI()

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Asynchronous function to generate text
async def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define asynchronous endpoint for text generation
@app.post("/generate/")
async def generate(input_text: List[str]):
    responses = []
    for prompt in input_text:
        try:
            response = await generate_text(prompt)  # Await asynchronous function
            responses.append({"prompt": prompt, "generated_text": response})
        except Exception as e:
            responses.append({"prompt": prompt, "error": str(e)})
    return responses
