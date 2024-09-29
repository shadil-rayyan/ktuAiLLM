import os
import streamlit as st
from dotenv import load_dotenv
import pinecone
from utils.pinecone_handler import initialize_pinecone, query_pinecone
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = initialize_pinecone(PINECONE_API_KEY)
index_name = "ktuai"
index = pc.Index(index_name)  # Updated to use the correct initialization

# Function to handle user questions
def user_input(user_question, subject, semester, department):
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    user_embedding = embeddings_model.embed_documents([user_question])[0]
    relevant_chunks = query_pinecone(index, user_embedding)

    # Filter chunks by subject and semester metadata
    filtered_chunks = [
        match for match in relevant_chunks["matches"] if
        match["metadata"]["subject"] == subject and
        match["metadata"]["semester"] == semester and
        match["metadata"]["department"] == department
    ]

    if filtered_chunks:
        # Create response using relevant chunks
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        response = model.generate_response({"input_documents": filtered_chunks, "question": user_question})
        return response["output_text"]
    else:
        return "No relevant information found."

def main():
    st.set_page_config(page_title="AI Subject Assistant with PDF")
    st.header("AI Subject Assistant")

    # Sidebar for selecting department, semester, and subject
    with st.sidebar:
        st.title("Select Options:")
        department = st.selectbox("Department", ["CS", "IT", "ECE", "ME", "Civil"])
        semester = st.selectbox("Semester", ["1", "2", "3", "4", "5", "6", "7", "8"])
        subject = st.selectbox("Subject", ["Math", "Physics", "CS101", "Networks", "Algorithms"])

    # User input for asking questions
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        answer = user_input(user_question, subject, semester, department)
        st.write("Reply: ", answer)

if __name__ == "__main__":
    main()
