import streamlit as st

# Create a sidebar with navigation
st.sidebar.title("RAG Experience")
page = st.sidebar.selectbox("Go to", ["PDF Uploader", "RAG Chat"])

# Import the pages
if page == "PDF Uploader":
    from src import pdf_frontend
    pdf_frontend.run()
elif page == "RAG Chat":
    from src import simple_frontend
    simple_frontend.run()
