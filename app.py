# chat_app.py
import os
import sys

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    raise RuntimeError("pysqlite3 is not installed. Make sure it's in requirements.txt")

import json
import chromadb
import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from chromadb.errors import NotFoundError
import asyncio

if sys.platform.startswith('win'):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


# --- Load API Key from config.json ---
def load_api_key(config_path="config.json"):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config.get("GROQ_API_KEY")

GROQ_API_KEY = load_api_key()

# --- Initialize ChromaDB & Embedding Model ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    collection = chroma_client.get_collection(name="ai_knowledge_base")
except NotFoundError:
    collection = chroma_client.create_collection(name="ai_knowledge_base")


# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# --- Chunk & Upsert to ChromaDB ---
def chunk_and_upsert(document_text, chunk_size=200, chunk_overlap=50, batch_size=10):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(document_text)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = [embedding_model.embed_query(chunk) for chunk in batch]
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=[f"doc_chunk_{i+j}" for j in range(len(batch))],
            metadatas=[{"chunk_index": i+j} for j in range(len(batch))]
        )
    return f"Upserted {len(chunks)} chunks to the database."

# --- Ingest PDF (only if run directly) ---
if __name__ == "__main__":
    pdf_path = "resume.pdf"
    if not os.path.exists(pdf_path):
        print(f"⚠️ PDF file not found at: {pdf_path}")
    else:
        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            result = chunk_and_upsert(text)
            print(result)
        else:
            print("⚠️ No text found in the PDF!")

# --- Initialize Chat Model & Memory ---
memory = ConversationBufferMemory(return_messages=True)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", api_key=GROQ_API_KEY)

# --- Streamlit Page Config ---
st.set_page_config(page_title="ChatWithMe", page_icon="🤖", layout="wide")
st.title("Chat With Ritesh")
st.write("Your personal chat bot....")

# --- Retrieve Context ---
def retrieve_context(query, top_k=1):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]

# --- Evaluate Response Similarity (Optional) ---
def evaluate_response(user_query, bot_response, context):
    response_embedding = semantic_model.encode(bot_response, convert_to_tensor=True)
    context_embedding = semantic_model.encode(context, convert_to_tensor=True)
    return util.pytorch_cos_sim(response_embedding, context_embedding)[0][0].item()

# --- Query AI Model ---
def query_llama3(user_query):
    system_prompt = """
## **Objective**  
The chatbot will serve as an interactive assistant based on Ritesh Chougule's background, technical skills, projects, and certifications. It should deliver:
-Quick, friendly responses for simple queries.
-In-depth, professional explanations for complex topics, backed by insights from Ritesh’s resume.

## **Response Styles**  
1. **Concise Responses (Simple Queries):**  
   - Under six words.  
   - Use relevant emojis (e.g., 😊, 🚀, 👍) to convey friendliness and clarity.  

2. **Detailed Explanations (In-Depth Queries):**    
   - Provide a warm, empathetic, and structured explanation.  
   - Always maintain precision and clarity, referencing relevant data from the PDF.  

3. **Tone & Style:**  
   - Human-like, empathetic, and helpful.  
   - Avoid revealing any AI or non-human nature.  
   - Adapt answer length based on the query’s complexity.

## **Warnings**  
- Never disclose any AI-related details or mention you’re a system.  
- Remain consistent with Ritesh Chougule's persona—an expert Android developer with advanced problem-solving skills.  
- Ensure all references to the PDF are correct and accurately represent Ritesh's background.

## **Context Dump**  
- **Primary Source (Uploaded PDF):** Contains comprehensive details on Ritesh Chougule's education, technical skills, projects, certifications, and overall professional expertise.  
- **Key Capabilities:**  
  - Dynamically adapt response length (short vs. long) based on the user’s query.  
  - Insert emojis to add warmth and clarity.  
  - Provide real-world applicability in every solution.  
- **Behavioral Directives:**  
  - Present yourself as a knowledgeable human expert.  
  - Never break the “no AI disclosure” rule.  
  - Prioritize user understanding and real-world context.
"""

    retrieved_context = retrieve_context(user_query)
    messages = [SystemMessage(content=system_prompt)] + memory.chat_memory.messages + [HumanMessage(content=user_query)]

    try:
        response = chat.invoke(messages)
        return response.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- Streamlit Chat UI ---
for msg in memory.chat_memory.messages:
    role = "user" if msg.type == "human" else "assistant"
    st.chat_message(role).write(msg.content)

user_input = st.chat_input("Type your message...")

if user_input:
    memory.chat_memory.add_user_message(user_input)
    ai_response = query_llama3(user_input)
    memory.chat_memory.add_ai_message(ai_response)
    st.chat_message("assistant").write(ai_response)