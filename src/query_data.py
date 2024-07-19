import argparse
from langchain_community.vectorstores.lancedb import LanceDB
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lancedb import connect
from src.vector_db import Embedding_Vector
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv('openai_key')

LANCE_PATH = "data/lancedb"
PROMPT_TEMPLATE = """
    Answer the question based ONLY on the following context:

    {context}

    ---

    Answer the question based ONLY on the above context: {question}

"""
class LLM_Rag:

    def __init__(self, prompt_template:str, lance_path:str, openai_key:str):
        self.prompt_template = prompt_template
        self.lance_path = lance_path
        self.openai_key = openai_key

    def query_rag(self, query_text: str):
        con = connect(self.lance_path)
        ev = Embedding_Vector(openai_key=self.openai_key, path_db='data/.lancedb')
        db = LanceDB(connection=con, embedding= ev.get_embedding_function())

        results = db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = ev.model_openai()
        response_text = model.invoke(prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return response_text, formatted_response

if __name__ == '__main__':
    llm = LLM_Rag(prompt_template=PROMPT_TEMPLATE, lance_path='data/.lancedb')
    response, fr= llm.query_rag('O que é verossimilhança?')
    print(fr)
