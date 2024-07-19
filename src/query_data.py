import argparse
from langchain_community.vectorstores.lancedb import LanceDB
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lancedb import connect
from src.vector_db import get_embedding_function, model_openai
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv('openai-key')

LANCE_PATH = "data/lancedb"


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}


"""

async def query_rag(query_text: str):
    con = connect(LANCE_PATH)
    db = LanceDB(connection=con, embedding=get_embedding_function(openai_key=key))

    results = db.asimilarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = model_openai(key=key)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return await response_text, formatted_response

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    query_text = "How to determine the linear correlation by some variables?"
    response, _ = query_rag(query_text)
    print(response.content)