from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncIterator
import asyncio
import io
import os
from dotenv import load_dotenv

from src.query_data import LLM_Rag

load_dotenv()
key = os.getenv('openai-key')

lance_path = "data/.lancedb"
prompt_template = """
    Answer the question based ONLY on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}

"""


app = FastAPI()

#prompt_template = 