import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

load_dotenv()
key = os.getenv('openai-key')

def load_documents(path:str):
    doc_loader = PyPDFDirectoryLoader(path)
    return doc_loader.load()

def split_documents(documents:list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function  = len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

if __name__ == '__main__':
    islp_pdf = 'data'
    documents = load_documents(islp_pdf)
    print(documents[0])

# def generate_text(prompt: str):
#     response = llm.generate([prompt])
#     return response[0]['text']

# # Example usage
# prompt = "Tell me a story about a brave knight."
# response = generate_text(prompt)
# print(response)
