import os
import numpy as np
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import pdfplumber
from pinecone import Pinecone
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
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "ktuai"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,  # Assuming 768 is the dimension of embeddings
            metric='euclidean',  # Specify the metric for similarity search
        )
    return pc.Index(index_name)

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Check if page text is not None
                    text += page_text + "\n"  # Add newline to separate pages
    return text

# Function to chunk the raw text into specified chunk size
def chunk_text(raw_text, chunk_size=1000):
    words = raw_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

# Function to generate a hash for a given text chunk
def generate_chunk_id(text_chunk):
    return hashlib.md5(text_chunk.encode('utf-8')).hexdigest()

# Function to check if the chunk is already in Pinecone
def is_chunk_indexed(index, chunk_id):
    try:
        # Try to fetch the chunk by ID
        result = index.fetch([chunk_id])
        return chunk_id in result['vectors']
    except Exception as e:
        logging.error(f"Error checking chunk ID in Pinecone: {e}")
        return False

# Normalize embedding vector
def normalize_vector(vector):
    """Normalize the embedding vector and handle edge cases."""
    vector = np.array(vector)  # Ensure it's a numpy array
    norm = np.linalg.norm(vector)
    if norm == 0:  # Handle zero vectors
        logging.warning("Encountered zero vector, returning unmodified.")
        return vector.tolist()  # Returning as a list to match expected output
    return (vector / norm).tolist()

# Create embeddings
def create_embeddings(text_chunks):
    """Create embeddings from text chunks and normalize them."""
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = embeddings_model.embed_documents(text_chunks)
    normalized_embeddings = [normalize_vector(emb) for emb in embeddings]

    logging.info(f"Generated {len(normalized_embeddings)} normalized embeddings.")
    for i, emb in enumerate(normalized_embeddings):
        if not isinstance(emb, list) or len(emb) != 768 or any(np.isnan(emb)) or any(np.isinf(emb)):
            logging.warning(f"Invalid embedding at index {i}: {emb}")
        else:
            logging.info(f"Embedding {i} has valid length: {len(emb)}")

    return normalized_embeddings

# Validate that embeddings are numeric and have the correct dimension
def validate_embeddings(embeddings):
    """Validate embeddings to ensure they are numeric and of correct dimensions."""
    for i, emb in enumerate(embeddings):
        if not isinstance(emb, list) or len(emb) != 768 or any(np.isnan(emb)) or any(np.isinf(emb)):
            logging.warning(f"Invalid embedding at index {i}: {emb}")
            return False
    return True

# Store embeddings in Pinecone
def store_embeddings_in_pinecone(index, text_chunks, embeddings, subject, semester, department):
    for i, (text, emb) in enumerate(zip(text_chunks, embeddings)):
        chunk_id = generate_chunk_id(text)
        logging.info(f"Uploading chunk ID: {chunk_id} with embedding: {emb}")
        if not is_chunk_indexed(index, chunk_id):
            try:
                index.upsert([(chunk_id, emb, {"text": text, "subject": subject, "semester": semester, "department": department})])
                logging.info(f"Successfully upserted chunk ID: {chunk_id}")
            except Exception as e:
                logging.error(f"Failed to upsert chunk ID: {chunk_id} with error: {e}")
        else:
            logging.info(f"Chunk {chunk_id} is already indexed.")

# Process a single PDF and upload it to Pinecone
def process_pdf(pdf_path, department, semester, subject, index):
    try:
        logging.info(f"Processing PDF: {pdf_path}")
        raw_text = get_pdf_text([pdf_path])
        text_chunks = chunk_text(raw_text)
        embeddings = create_embeddings(text_chunks)

        # Validate embeddings before uploading
        if not validate_embeddings(embeddings):
            logging.error(f"Invalid embeddings generated for PDF: {pdf_path}.")
            return

        store_embeddings_in_pinecone(index, text_chunks, embeddings, subject, semester, department)
        logging.info(f"Successfully processed and uploaded: {pdf_path}")
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path}: {e}")

# Find and process all PDFs with parallelism
def process_and_upload_pdfs():
    base_path = "pdfs"
    pdf_tasks = []
    index = initialize_pinecone_and_index()

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
            executor.submit(process_pdf, pdf_path, department, semester, subject, index)
            for pdf_path, department, semester, subject in pdf_tasks
        ]
        for future in futures:
            future.result()  # Wait for the task to complete

    try:
        count = index.describe_index_stats()['total_vector_count']
        logging.info(f"Total vectors in Pinecone index after upload: {count}")
    except Exception as e:
        logging.error(f"Error retrieving index stats: {e}")

if __name__ == "__main__":
    process_and_upload_pdfs()
