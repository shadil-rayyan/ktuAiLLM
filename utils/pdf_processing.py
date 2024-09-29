import pdfplumber
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Check if page text is not None
                    text += page_text + "\n"  # Add newline to separate pages
    return text

def chunk_text(raw_text, chunk_size=1000):
    # Splitting text into chunks of a specified size
    words = raw_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

def create_embeddings(text_chunks):
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings_model.embed_documents(text_chunks)
