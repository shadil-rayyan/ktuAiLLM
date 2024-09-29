import os
from dotenv import load_dotenv
from utils.pdf_processing import get_pdf_text, chunk_text, create_embeddings
from utils.pinecone_handler import initialize_pinecone, create_index_if_not_exists, store_embeddings_in_pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = initialize_pinecone(PINECONE_API_KEY)
index_name = "ktuai"

# Create index if it doesn't exist
create_index_if_not_exists(pc, index_name)

# Function to find and process all PDFs in the department and semester directories
def process_and_upload_pdfs():
    base_path = "pdfs"

    # Loop through each department and semester folder
    for department in os.listdir(base_path):
        department_path = os.path.join(base_path, department)
        if not os.path.isdir(department_path):
            continue
        
        for semester in os.listdir(department_path):
            semester_path = os.path.join(department_path, semester)
            if not os.path.isdir(semester_path):
                continue
            
            # Process PDFs in the current department and semester
            pdf_files = []
            for subject in os.listdir(semester_path):
                subject_path = os.path.join(semester_path, subject)
                if not os.path.isdir(subject_path):
                    continue
                
                for filename in os.listdir(subject_path):
                    if filename.endswith('.pdf'):
                        pdf_files.append(os.path.join(subject_path, filename))
            
            if not pdf_files:
                print(f"No PDFs found for Department: {department}, Semester: {semester}.")
                continue
            
            # Extract text and chunk it
            raw_text = get_pdf_text(pdf_files)
            print(f"Extracted text from PDFs: {raw_text[:200]}...")  # Print the first 200 characters

            text_chunks = chunk_text(raw_text)
            print(f"Number of text chunks created: {len(text_chunks)}")  # Print the number of chunks

            embeddings = create_embeddings(text_chunks)
            print(f"Embeddings shape: {[len(embedding) for embedding in embeddings]}")  # Print the shape of each embedding

            # Store embeddings in Pinecone with metadata for each PDF
            for pdf_file in pdf_files:
                subject_name = os.path.basename(os.path.dirname(pdf_file))  # Get the subject from the directory structure
                store_embeddings_in_pinecone(pc, index_name, text_chunks, embeddings, subject_name, semester, department)

            print(f"Processing and uploading completed for Department: {department}, Semester: {semester}.")

    # Check the total number of vectors in Pinecone
    count = pc.Index(index_name).describe_index_stats()['total_vector_count']
    print(f"Total vectors in Pinecone index '{index_name}' after upload: {count}")

if __name__ == "__main__":
    process_and_upload_pdfs()
