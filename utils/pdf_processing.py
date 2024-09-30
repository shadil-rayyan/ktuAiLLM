import pdfplumber
import hashlib
import pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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
        print(f"Error checking chunk ID in Pinecone: {e}")
        return False

# Function to create and store embeddings, only for new chunks
def create_embeddings(index, text_chunks):
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_chunks = []

    for chunk in text_chunks:
        chunk_id = generate_chunk_id(chunk)
        
        if not is_chunk_indexed(index, chunk_id):
            # If the chunk is not indexed, create the embedding and store it
            embedding = embeddings_model.embed_documents([chunk])[0]  # Assume batch size of 1
            index.upsert([(chunk_id, embedding, {"text": chunk})])  # Add metadata as needed
            new_chunks.append(chunk_id)
        else:
            print(f"Chunk {chunk_id} is already indexed.")
    
    return new_chunks

# Example usage
def process_pdfs_and_store(pdf_docs, index):
    # Get the raw text from the PDF documents
    raw_text = get_pdf_text(pdf_docs)
    
    # Split the text into chunks
    text_chunks = chunk_text(raw_text)
    
    # Create embeddings and store them, avoiding duplicates
    create_embeddings(index, text_chunks)
