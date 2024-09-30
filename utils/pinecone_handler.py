import os
import pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def initialize_pinecone(api_key):
    """Initialize Pinecone with the provided API key."""
    pinecone.init(api_key=api_key)

def create_index_if_not_exists(index_name):
    """Create the Pinecone index if it does not already exist."""
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

def store_embeddings_in_pinecone(index_name, text_chunks, embeddings, subject, semester, department):
    """Store embeddings into the Pinecone index."""
    index = pinecone.Index(index_name)
    upsert_data = []
    for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        metadata = {
            "subject": subject,
            "semester": semester,
            "department": department,
            "page_number": i
        }
        upsert_data.append((f"chunk-{i}", embedding, metadata))
    
    index.upsert(upsert_data)

def query_pinecone(index, user_embedding, top_k=5):
    """Query Pinecone to find the top K most similar chunks."""
    response = index.query(queries=[user_embedding], top_k=top_k, include_metadata=True)
    return response
