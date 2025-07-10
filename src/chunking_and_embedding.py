import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import os


df = pd.read_csv("../data/filtered_complaints12.csv")

# Initialize Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

# Load Sentence Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

#Prepare ChromaDB Client
CHROMA_DIR = "../vector_store"
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="complaints")




# Chunk, Embed, and Index 
doc_id = 0  # unique ID for Chroma
for idx, row in df.iterrows():
    complaint_text = row["cleaned_Consumer_complaint_narrative"]
    metadata = {
        "complaint_id": str(row["Complaint ID"]),
        "product": row["Product"]
    }

    # Split into chunks
    chunks = text_splitter.split_text(complaint_text)

    # Embed and add each chunk to ChromaDB
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()

        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_index"] = i
        chunk_metadata["text"] = chunk

        collection.add(
            ids=[f"doc-{doc_id}"],
            embeddings=[embedding],
            metadatas=[chunk_metadata],
            documents=[chunk]
        )
        doc_id += 1

client.persist()