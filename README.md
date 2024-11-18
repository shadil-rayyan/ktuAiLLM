# Chat with KTU Notes Using Gemini 💁

This Streamlit application enables KTU (Kerala Technological University) BTech students to interact with various types of academic documents, such as textbooks, notebooks, and PDFs. Powered by **Google's Gemini generative AI** and a **Retrieval-Augmented Generation (RAG) model**, this tool provides accurate answers to student questions based on the content of the uploaded documents.

## Features

- **Upload Academic Documents**: Users can upload PDFs, textbooks, notebooks, and other academic materials.
- **Process Documents**: Extracts relevant text from the uploaded files and creates text chunks for indexing.
- **Search and Chat**: Users can ask questions related to the uploaded documents and receive answers using FAISS vector search combined with the power of Gemini's conversational AI.
- **Improved Accuracy**: The system is designed to select the most relevant document (e.g., textbook, notebook) to answer questions more accurately using the RAG model.

## Requirements

- Python 3.7+
- Libraries specified in `requirements.txt`
- **Google API Key** configured via the environment variable `GOOGLE_API_KEY`
- **FAISS** for vector search

## Installation and Setup

1. **Clone the repository**:

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:

    Create a `.env` file in the root directory and add your Google API Key:

    ```makefile
    GOOGLE_API_KEY=<your_google_api_key>
    ```

## Usage

1. **Run the application**:

    ```bash
    streamlit run app.py
    ```

2. **Interact with the application**:

    - Upload academic documents (textbooks, notebooks, PDFs) via the sidebar.
    - Click "Submit & Process" to extract text and create vector embeddings from the documents.
    - Ask a question related to the content of the documents using the text input field.
    - Click "Ask" to receive an answer generated by the **Gemini conversational AI** model based on the selected document (textbook, notebook, etc.).

## About

This project demonstrates the integration of **Retrieval-Augmented Generation (RAG)** for accurate question answering using **Google's Gemini API**. It’s tailored for KTU BTech students to quickly get precise answers from academic materials (textbooks, notes, etc.) without manually searching through large documents.

The system combines the power of **FAISS vector search** to find the most relevant text chunks and **Gemini's generative AI** to generate coherent and accurate answers.

## Built Upon

This project is built upon an earlier repository that demonstrated the use of the **RAG model** with Gemini for answering questions from PDFs. The underlying RAG architecture allows the system to intelligently select the most relevant document (e.g., textbook, notebook) for better accuracy when answering student queries.

## Credits

- **Streamlit**: Frontend framework for building interactive web applications.
- **Google Gemini**: Used for generative AI-powered conversational model.
- **FAISS**: Utilized for semantic vector search to retrieve relevant document chunks.

---
