import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Load vectorstore from disk
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("ramayana_index", embeddings, allow_dangerous_deserialization=True)


# Load Groq LLM
@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="LLaMA3-8b-8192")

from langchain.prompts import PromptTemplate

def build_prompt():
    return PromptTemplate(
        input_variables=["question", "context"],
        template="""
You are a factual reasoning assistant trained on the Valmiki Ramayana. Your job is to **verify a given statement strictly using the provided context**.

Follow these rules exactly:

1. ✅ If the context **clearly supports** the statement:
   - Identify the **top matching verse** that best proves the statement.
   - Include **up to 2 additional supporting verses** as secondary evidence.
   - Output format:

✅ True, Reference:

"The context clearly supports the statement:

Primary Reference:  
<Kanda/Book>, Chapter <chapter>, Verse <verse>: "<top matching translation>"

Also mentioned in:  
- <Kanda/Book>, Chapter <chapter>, Verse <verse>: "<supporting translation>"  
- <Kanda/Book>, Chapter <chapter>, Verse <verse>: "<supporting translation>"

The above verses confirm that the statement is ✅ True."

### Examples:

#### ✅ True  
STATEMENT: Rama is the eldest son of King Dasharatha.  
CONTEXT:  
1. KishkindaKanda, Chapter 4, Verse 8: "this one is his eldest son, and he is renowned among people by the name of Rama..."  
2. AranyaKanda, Chapter 17, Verse 15: "I am his eldest son, and people hear of me by name Rama."  
3. KishkindaKanda, Chapter 62, Verse 4: "to him there will be a great-resplendent a son who will be known as Rama..."

✅ True, Reference:

The context clearly supports the statement:

Primary Reference:  
KishkindaKanda, Chapter 4, Verse 8: "this one is his eldest son, and he is renowned among people by the name of Rama..."

Also mentioned in:  
- AranyaKanda, Chapter 17, Verse 15: "I am his eldest son, and people hear of me by name Rama."  
- KishkindaKanda, Chapter 62, Verse 4: "to him there will be a great-resplendent a son who will be known as Rama..."

The above verses confirm that the statement is ✅ True.

---

2. ❌ If the context **clearly contradicts** the statement:
   - Output: ❌ False  
#Examples:Hanuman was the son of Ravana.
          Rama’s exile lasted only one year rather than the traditionally recounted fourteen years.
3. ⚠️ If the context is **irrelevant or insufficient**:
   - Output: ⚠️ Irrelevant  
#Examples: Hanuman was in Avegers.
**Important Instructions**:
- Never assume or guess. Use only the text in the context.
- If multiple verses support the same idea, rank them and show the top match first.
- Use exact translations from context. Maintain Ramayana terminology (e.g., Kanda, Chapter, Verse).

---



Now verify the following:

STATEMENT: {question}

CONTEXT:  
{context}
"""
    )

# Inference logic
def classify_statement(question, vectorstore, llm):
    retriever = vectorstore.as_retriever(search_type="similarity", k=4)
    docs = retriever.get_relevant_documents(question)

    context = ""
    for doc in docs:
        meta = doc.metadata
        context += f"Kanda/Book: {meta['Kanda/Book']}, Chapter: {meta['Sarga/Chapter']}, Verse: {meta['Shloka/Verse']}\n"
        context += f"Translation: {doc.page_content}\n\n"

    chain = LLMChain(llm=llm, prompt=build_prompt())
    return chain.run({"question": question, "context": context})


# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="Ramayana Fact Checker", page_icon="🕉️")
st.title("🕉️ Ramayana Fact Checker")

question = st.text_input("🔍 Enter a statement to verify:").strip()
if question:
    with st.spinner("🔍 Searching the Ramayana..."):
        vs = load_vectorstore()
        llm = get_llm()
        result = classify_statement(question, vs, llm)

        st.markdown("### 🧾 Result")
        if result.startswith("✅"):
            st.success(result)
        elif result.startswith("❌"):
            st.error(result)
        elif result.startswith("⚠️"):
            st.warning(result)
        else:
            st.info(result)
