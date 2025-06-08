import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Load dataset
df = pd.read_csv("RamayanDataSet.csv", encoding='ISO-8859-1')

# Prepare documents with metadata
docs = []
for _, row in df.iterrows():
    content = row["English Translation"]
    metadata = {
        "Kanda/Book": row["Kanda/Book"],
        "Sarga/Chapter": row["Sarga/Chapter"],
        "Shloka/Verse": row["Shloka/Verse"]
    }
    docs.append(Document(page_content=str(content), metadata=metadata))

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Save to disk
vectorstore.save_local("ramayana_index")
print("✅ Vector store saved to 'ramayana_index/'")
