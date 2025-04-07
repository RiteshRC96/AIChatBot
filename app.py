import os
import sys
import chromadb
import PyPDF2
import streamlit as st
import numpy as np
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util

# Initialize Persistent ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = chroma_client.get_collection(name="ai_knowledge_base")
except chromadb.errors.InvalidCollectionException:
    collection = chroma_client.create_collection(name="ai_knowledge_base")

# Initialize Embedding & Chat Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
chat = ChatGroq(
    temperature=0.7,
    model_name="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Memory
memory = ConversationBufferMemory(return_messages=True)

# Streamlit Page Configuration
st.set_page_config(page_title="ChatWithMe", page_icon="🤖", layout="centered")
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
        }
        .chat-container {
            max-width: 700px;
            margin: auto;
            padding: 20px;
        }
        .user-msg {
            background-color: #0084ff;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: right;
        }
        .bot-msg {
            background-color: #e5e5ea;
            color: black;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: left;
        }
        .message-box {
            width: 100%;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Chat With Ritesh")
st.write("Your personal AI assistant! 🚀")

# Function to Retrieve Context
def retrieve_context(query, top_k=1):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results.get("documents") else ["No relevant context found."]

# Function to Query Llama3
def query_llama3(user_query):
    """
    1. Gather system prompt, memory, and new user query.
    2. Retrieve additional context from ChromaDB.
    3. Pass all messages to the LLM.
    4. Return the AI's response.
    """
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
    messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=f"## Retrieved Context:\n{retrieved_context}"),
        *memory.chat_memory.messages,
        HumanMessage(content=user_query)
    ]
    try:
        response = chat.invoke(messages)
        return response.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Chat UI
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
with st.expander("💬 Chat History", expanded=False):
    for msg in memory.chat_memory.messages:
        if msg.type == "human":
            st.markdown(f"<div class='user-msg'>{msg.content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{msg.content}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# User Input
user_input = st.text_input("Type your message...")
if user_input:
    memory.chat_memory.add_user_message(user_input)
    ai_response = query_llama3(user_input)
    memory.chat_memory.add_ai_message(ai_response)
    st.experimental_rerun()
