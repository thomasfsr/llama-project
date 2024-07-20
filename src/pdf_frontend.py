import streamlit as st
import os
import glob
from time import sleep

def run():
    uploaded_files = st.file_uploader("Insert PDFs you want to query.", 
                                    type="pdf",
                                    accept_multiple_files=True,
                                    )
    button_show = st.button("Show saved files")

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            #st.success(f"Saved file: {uploaded_file.name}")

    if button_show:
        saved_files = glob.glob("data/*.pdf")
        if saved_files:
            st.write("PDFs saved in the 'data' directory:")
            for saved_file in saved_files:
                st.success(saved_file)
        else:
            st.write("No files saved yet.")