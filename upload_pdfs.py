import os
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from utils.pdf_processing import get_pdf_text, chunk_text
from utils.pinecone_handler import (
    initialize_pinecone,
    create_index_if_not_exists,
    store_embeddings_in_pinecone,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure logging
logging.basicConfig(
    filename="upload.log", level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone and create index
def initialize_pinecone_and_index():
    pc = initialize_pinecone(PINECONE_API_KEY)
    index_name = "ktuai"
    create_index_if_not_exists(index_name)
    return pc, index_name

# Normalize embedding vector
def normalize_vector(vector):
    """Normalize the embedding vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:  # Handle zero vectors
        return vector
    return (vector / norm).tolist()

# Create embeddings
def create_embeddings(text_chunks):
    """Create embeddings from text chunks and normalize them."""
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = embeddings_model.embed_documents(text_chunks)
    normalized_embeddings = [normalize_vector(emb) for emb in embeddings]

    logging.info(f"Generated {len(normalized_embeddings)} normalized embeddings.")
    for i, emb in enumerate(normalized_embeddings):
        if not isinstance(emb, list) or len(emb) != 768:
            logging.warning(f"Invalid embedding at index {i}: {emb}")
        else:
            logging.info(f"Embedding {i} has valid length: {len(emb)}")

    return normalized_embeddings

# Process a single PDF and upload it to Pinecone
def process_pdf(pdf_path, department, semester, subject, pc, index_name):
    try:
        logging.info(f"Processing PDF: {pdf_path}")
        raw_text = get_pdf_text([pdf_path])  # Ensure this function works as expected
        text_chunks = chunk_text(raw_text)  # Ensure this function works as expected
        embeddings = create_embeddings(text_chunks)

        # Validate embeddings before uploading
        invalid_embeddings = [e for e in embeddings if not isinstance(e, list) or len(e) != 768]
        if invalid_embeddings:
            logging.error(f"Invalid embeddings generated for PDF: {pdf_path}. Found {len(invalid_embeddings)} invalid embeddings.")
            return

        store_embeddings_in_pinecone(index_name, text_chunks, embeddings, subject, semester, department)
        logging.info(f"Successfully processed and uploaded: {pdf_path}")
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path}: {e}")

# Find and process all PDFs with parallelism
def process_and_upload_pdfs():
    base_path = "pdfs"
    pdf_tasks = []
    pc, index_name = initialize_pinecone_and_index()

    for department in os.listdir(base_path):
        department_path = os.path.join(base_path, department)
        if not os.path.isdir(department_path):
            continue
        
        for semester in os.listdir(department_path):
            semester_path = os.path.join(department_path, semester)
            if not os.path.isdir(semester_path):
                continue
            
            for subject in os.listdir(semester_path):
                subject_path = os.path.join(semester_path, subject)
                if not os.path.isdir(subject_path):
                    continue
                
                for filename in os.listdir(subject_path):
                    if filename.endswith('.pdf'):
                        pdf_path = os.path.join(subject_path, filename)
                        pdf_tasks.append((pdf_path, department, semester, subject))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_pdf, pdf_path, department, semester, subject, pc, index_name)
            for pdf_path, department, semester, subject in pdf_tasks
        ]
        for future in futures:
            future.result()  # Wait for the task to complete

    try:
        count = pc.Index(index_name).describe_index_stats()['total_vector_count']
        logging.info(f"Total vectors in Pinecone index '{index_name}' after upload: {count}")
    except Exception as e:
        logging.error(f"Error retrieving index stats: {e}")

if __name__ == "__main__":
    process_and_upload_pdfs()
