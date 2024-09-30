import os
import numpy as np
import logging
from dotenv import load_dotenv
import pdfplumber
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
def initialize_pinecone(api_key):
    """Initialize Pinecone with the provided API key."""
    return Pinecone(api_key=api_key)

# Create index if it doesn't exist
def create_index_if_not_exists(pc, index_name="ktuai"):
    """Create the Pinecone index if it does not already exist."""
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,  # Adjust based on your embeddings dimension
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc.Index(index_name)

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    """Extract text from a list of PDF documents."""
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
    """Chunk the raw text into smaller segments."""
    words = raw_text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Store embeddings in Pinecone
def store_embeddings_in_pinecone(index, text_chunks, embeddings, subject, semester, department):
    """Store embeddings into the Pinecone index."""
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

# Function to check if an embedding is valid
def is_valid_vector(embedding, expected_dim=768):
    """Check if the given embedding is a valid vector."""
    return (
        isinstance(embedding, list) and
        len(embedding) == expected_dim and
        all(isinstance(x, float) and -1 <= x <= 1 for x in embedding)  # Ensure values are within a reasonable range
    )

# Query Pinecone
def query_pinecone(index, embedding, top_k=5):
    """Query Pinecone index with the given embedding and return top_k results."""
    try:
        query_response = index.query(queries=[embedding], top_k=top_k, include_metadata=True)
        return query_response
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return None

def user_input(user_question, subject, semester, department, index):
    """Process the user question and return an AI-generated answer."""
    try:
        # Create embeddings for the user question
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        user_embedding = embeddings_model.embed_documents([user_question])[0]

        # Log the generated embedding for debugging
        logging.info(f"Generated embedding: {user_embedding}")

        # Convert to float32 and check for NaN or Infinite values
        user_embedding = np.array(user_embedding, dtype=np.float32).tolist()
        if np.any(np.isnan(user_embedding)) or np.any(np.isinf(user_embedding)):
            logging.error(f"Embedding contains NaN or Infinite values: {user_embedding}")
            return "Error: Embedding contains invalid values."

        # Validate the embedding before querying
        if not is_valid_vector(user_embedding):
            logging.error(f"Invalid embedding vector for question: {user_question}")
            return "Error: Invalid embedding vector generated."

        # Query the index
        relevant_chunks = query_pinecone(index, user_embedding)

        # Filter chunks by subject, semester, and department metadata
        if relevant_chunks is None:
            return "An error occurred while querying the index."

        filtered_chunks = [
            match for match in relevant_chunks["matches"]
            if (match["metadata"].get("subject") == subject and
                match["metadata"].get("semester") == semester and
                match["metadata"].get("department") == department)
        ]

        # Generate response using the AI model if relevant chunks are found
        if filtered_chunks:
            model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
            response = model.generate_response({"input_documents": filtered_chunks, "question": user_question})
            return response.get("output_text", "No output text returned.")
        else:
            return "No relevant information found."
    except Exception as e:
        logging.error(f"Error during query: {e}")
        return "An error occurred while processing your request."

# Streamlit UI
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="AI Subject Assistant with PDF")
    st.header("AI Subject Assistant")

    # Initialize Pinecone and create index if it doesn't exist
    pc = initialize_pinecone(PINECONE_API_KEY)
    index = create_index_if_not_exists(pc)

    # Sidebar for selecting department, semester, and subject
    with st.sidebar:
        st.title("Select Options:")
        department = st.selectbox("Department", ["CS", "IT", "ECE", "ME", "Civil"])
        semester = st.selectbox("Semester", [str(i) for i in range(1, 9)])  # Generates ["1", "2", ..., "8"]
        subject = st.selectbox("Subject", ["Math", "Physics", "CS101", "Networks", "Algorithms"])

    # User input for asking questions
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        answer = user_input(user_question, subject, semester, department, index)
        st.write("Reply: ", answer)

if __name__ == "__main__":
    main()
