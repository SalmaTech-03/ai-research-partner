# data_processing.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os

def process_uploaded_pdf(uploaded_file):
    """
    Loads a PDF, splits it into chunks, and returns the document chunks.
    Handles temporary file creation and cleanup.
    """
    if uploaded_file is None:
        return None
        
    # Create a temporary file to store the uploaded PDF to ensure compatibility
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    docs = None
    try:
        # Load the document from the temporary file path
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Split the document into smaller, more manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        docs = text_splitter.split_documents(documents)
        
    except Exception as e:
        print(f"Error processing PDF file: {e}")
        # We can also add a Streamlit error message here if needed,
        # but for now, we'll just return None.
        return None
    finally:
        # Clean up the temporary file after processing is complete
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
    
    return docs