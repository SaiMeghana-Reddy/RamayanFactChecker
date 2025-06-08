# RamayanFactChecker
IYD hackathon project which tells the statement provided is true , false or irrelevant according to ramayan.
# 🕉️ Ramayana Fact Checker - IYD Hackathon 2025

## 🔍 Overview

**Ramayana Fact Checker** is an AI-powered web application that verifies factual statements based on the English translations of verses from the Valmiki Ramayana. Built using LangChain, HuggingFace embeddings, and Streamlit, the app utilizes Retrieval-Augmented Generation (RAG) with a Groq-hosted LLaMA model to classify a statement as:

- ✅ **True** (with exact references)
- ❌ **False**
- ⚠️ **Irrelevant**

---

## 💡 Key Features

- 🔎 Retrieves contextually relevant Ramayana verses from a pre-built vector store.
- 🤖 Uses **LLaMA3-8B** hosted on **Groq** for inference via LangChain.
- 📖 Displays exact references (Kanda, Chapter, Verse) and translations to support the verdict.
- 📊 Uses sentence-transformer embeddings (`all-MiniLM-L6-v2`) for semantic understanding.

---

## 📁 File Structure

├── app.py # Main Streamlit application
├── vectorproduce.py # Script to convert Ramayana dataset into FAISS vectorstore
├── RamayanDataSet.csv # Dataset containing Ramayana English translations and metadata
├── .env # Stores GROQ_API_KEY
├── ramayana_index/ # Directory containing FAISS vectorstore files
└── README.md # This file

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

git clone https://github.com/SaiMeghana-Reddy/RamayanFactChecker.git
cd RamayanaFactChecker

#2. Create Virtual Environment

python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows

#3. Install Requirements

pip install -r requirements.txt
(If requirements.txt doesnt exist, create it using:)

pip freeze > requirements.txt

#4. Add Your Groq API Key

Create a .env file in the project root:

GROQ_API_KEY=your_actual_key_here

#5.🛠️ Build Vector Store (Run Once)

python vectorproduce.py

This will create and save a vector index (ramayana_index/) using FAISS.

#6.🚀 Run the Application

streamlit run app.py

Navigate to the URL shown in the terminal (e.g., http://localhost:8501).

✅ Example Usage
Input Statement:
csharp
Copy
Edit
Rama is the eldest son of King Dasharatha.
Output:
✅ True, Reference:

Primary Reference:
KishkindaKanda, Chapter 4, Verse 8: "this one is his eldest son, and he is renowned among people by the name of Rama..."

Also mentioned in:

AranyaKanda, Chapter 17, Verse 15

KishkindaKanda, Chapter 62, Verse 4

📚 Dataset Information
Source: [Valmiki Ramayana (English Translations)]

Fields: Kanda/Book, Chapter/Sarga, Shloka/Verse, English Translation

👨‍💻 Technologies Used
Streamlit: UI & Deployment

LangChain: LLM chaining and RAG pipeline

FAISS: Vector similarity search

HuggingFace: Sentence Transformers

Groq + LLaMA3-8B: High-speed factual reasoning

📌 Notes
Ensure ramayana_index/ is present before running app.py.

App caches both LLM and vectorstore for performance.

App supports real-time factual validation for up to k=4 nearest semantic matches.

You can experiment with different prompts in code for variety of outputs

🤝 Credits
Built by SaiMeghana Padala, NavyaSri and PavanKumar as part of IYD Hackathon 2025.

