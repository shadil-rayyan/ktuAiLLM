import os
import pinecone
from pinecone import ServerlessSpec

def initialize_pinecone(api_key):
    # Create an instance of the Pinecone class
    pc = pinecone.Pinecone(api_key=api_key)
    return pc

def create_index_if_not_exists(pc, index_name):
    # Check if the index exists
    if index_name not in pc.list_indexes().names():
        # Create the index with appropriate specifications
        pc.create_index(name=index_name, dimension=1536, metric='euclidean', 
                        spec=ServerlessSpec(cloud='aws', region='us-east-1'))

def store_embeddings_in_pinecone(pc, index_name, text_chunks, embeddings, subject, semester, department):
    # Access the index
    index = pc.Index(index_name)

    # Prepare the data to upsert
    upsert_data = []
    for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        metadata = {
            "subject": subject,
            "semester": semester,
            "department": department,
            "page_number": i
        }
        upsert_data.append((f"chunk-{i}", embedding, metadata))
    
    # Upsert embeddings into the Pinecone index
    index.upsert(upsert_data)

def query_pinecone(index, user_embedding, top_k=5):
    # Query Pinecone to find the top K most similar chunks
    response = index.query(queries=[user_embedding], top_k=top_k, include_metadata=True)
    return response
