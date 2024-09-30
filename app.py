import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import logging
from utils.pinecone_handler import initialize_pinecone, query_pinecone
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)
import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import logging
from utils.pinecone_handler import initialize_pinecone, query_pinecone
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

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
def initialize_pinecone_and_index(api_key):
    """Initialize Pinecone and return the index."""
    pc = initialize_pinecone(api_key)
    index_name = "ktuai"
    index = pc.Index(index_name)
    return index

# Create the index object
index = initialize_pinecone_and_index(PINECONE_API_KEY)

# Function to check if an embedding is valid
def is_valid_vector(embedding, expected_dim=768):
    """Check if the given embedding is a valid vector."""
    return (
        isinstance(embedding, list) and
        len(embedding) == expected_dim and
        all(isinstance(x, float) for x in embedding)
    )

# Function to handle user questions
def user_input(user_question, subject, semester, department):
    """Process the user question and return an AI-generated answer."""
    try:
        # Create embeddings for the user question
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        user_embedding = embeddings_model.embed_documents([user_question])[0]

        # Validate the embedding before querying
        if not is_valid_vector(user_embedding):
            logging.error(f"Invalid embedding vector for question: {user_question}")
            return "Error: Invalid embedding vector generated."

        # Convert to float32 and query Pinecone
        user_embedding = np.array(user_embedding).astype(np.float32).tolist()
        relevant_chunks = query_pinecone(index, user_embedding)

        # Filter chunks by subject, semester, and department metadata
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

    # Sidebar for selecting department, semester, and subject
    with st.sidebar:
        st.title("Select Options:")
        department = st.selectbox("Department", ["CS", "IT", "ECE", "ME", "Civil"])
        semester = st.selectbox("Semester", [str(i) for i in range(1, 9)])  # Generates ["1", "2", ..., "8"]
        subject = st.selectbox("Subject", ["Math", "Physics", "CS101", "Networks", "Algorithms"])

    # User input for asking questions
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        answer = user_input(user_question, subject, semester, department)
        st.write("Reply: ", answer)

if __name__ == "__main__":
    main()

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = initialize_pinecone(PINECONE_API_KEY)
index_name = "ktuai"
index = pc.Index(index_name)

# Function to check if an embedding is valid
def is_valid_vector(embedding, expected_dim=768):
    """Check if the given embedding is a valid vector."""
    return (
        isinstance(embedding, list) and
        len(embedding) == expected_dim and
        all(isinstance(x, float) for x in embedding)
    )

# Function to handle user questions
def user_input(user_question, subject, semester, department):
    """Process the user question and return an AI-generated answer."""
    try:
        # Create embeddings for the user question
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        user_embedding = embeddings_model.embed_documents([user_question])[0]

        # Validate the embedding before querying
        if not is_valid_vector(user_embedding):
            logging.error(f"Invalid embedding vector for question: {user_question}")
            return "Error: Invalid embedding vector generated."

        # Convert to float32 and query Pinecone
        user_embedding = np.array(user_embedding).astype(np.float32).tolist()
        relevant_chunks = query_pinecone(index, user_embedding)

        # Filter chunks by subject, semester, and department metadata
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

    # Sidebar for selecting department, semester, and subject
    with st.sidebar:
        st.title("Select Options:")
        department = st.selectbox("Department", ["CS", "IT", "ECE", "ME", "Civil"])
        semester = st.selectbox("Semester", [str(i) for i in range(1, 9)])  # Generates ["1", "2", ..., "8"]
        subject = st.selectbox("Subject", ["Math", "Physics", "CS101", "Networks", "Algorithms"])

    # User input for asking questions
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        answer = user_input(user_question, subject, semester, department)
        st.write("Reply: ", answer)

if __name__ == "__main__":
    main()
