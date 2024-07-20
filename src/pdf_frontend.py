import streamlit as st
import os
import glob
from dotenv import load_dotenv
load_dotenv()

key = os.getenv('openai_key')

def run(): 
    uploaded_files = st.file_uploader("Insert PDFs you want to query.", 
                                        type="pdf",
                                        accept_multiple_files=True,
                                        )
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)
    with col1:
        button_show = st.button("Show saved files")
        if button_show:
            saved_files = glob.glob("data/*.pdf")
            if saved_files:
                st.write("PDFs saved in the 'data' directory:")
                for saved_file in saved_files:
                    st.success(saved_file)
            else:
                st.write("No files saved yet.")

    with col2:
        button_load = st.button(':red[Load PDFs to LanceDB]')
        if button_load:
            from src.vector_db import Embedding_Vector
            motor = Embedding_Vector(key, 'data/.lancedb','data')
            motor.add_to_lancedb(chunk_size=800, chunk_overlap=80)