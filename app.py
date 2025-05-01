import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e2f;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #f0f0f0;
        padding: 10px;
    }
    
    h1, h3 {
        color: #00ffe5;
    }

    .chat-font {
        font-family: 'Segoe UI', sans-serif;
    }

    .stChatInput > div {
        background-color: #2b2b3d;
        border-radius: 10px;
        padding: 10px;
    }

    .user-msg, .bot-msg {
        padding: 12px 18px;
        border-radius: 15px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
        word-wrap: break-word;
        font-size: 16px;
    }

    .user-msg {
        background-color: #2b2b3d !important;
        color: #ffffff;
        border: 1px solid #5f5f87 !important;
        align-self: flex-end;
        margin-left: auto;
    }

    .bot-msg {
        background-color: #343449 !important;
        color: #e0e0e0;
        border: 1px solid #00ffe5 !important;
        align-self: flex-start;
        margin-right: auto;
    }

    .block-container {
        padding-top: 2rem;
    }

    /* Scrollbar styling for chat */
    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #2b2b3d;
    }

    ::-webkit-scrollbar-thumb {
        background: #444;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #666;
    }
</style>
""", unsafe_allow_html=True)




# Configure Google Gemini
genai.configure(api_key="AIzaSyBsq5Kd5nJgx2fejR77NT8v5Lk3PK4gbH8")  # Replace with your Gemini API key
# genai.configure(api_key=st.secrets["gemini_api_key"])
gemini = genai.GenerativeModel('gemini-1.5-flash')

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model

# Load data and create FAISS index
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('my_data.csv')  # Replace with your dataset file name
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("The CSV file must contain 'question' and 'answer' columns.")
            st.stop()
        df['context'] = df.apply(
            lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}", 
            axis=1
        )
        embeddings = embedder.encode(df['context'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])  # FAISS index for similarity search
        index.add(np.array(embeddings).astype('float32'))
        return df, index
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

# Load dataset and FAISS index
df, faiss_index = load_data()

# App Header
st.markdown('<h1 class="chat-font">🤖 Puji Clone Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="chat-font">Ask me anything, and I\'ll respond as Puji!</h3>', unsafe_allow_html=True)
st.markdown("---")

# Function to find the closest matching question using FAISS
def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=1)  # Top 1 match
    if I.size > 0:
        return df.iloc[I[0][0]]['answer']  # Return the closest answer
    return None

# Function to generate a refined answer using Gemini
def generate_refined_answer(query, retrieved_answer):
    prompt = f"""You are Puji, an AI, ML, and DL instructor. Respond to the following question in a friendly and conversational tone:
    Question: {query}
    Retrieved Answer: {retrieved_answer}
    - Provide a detailed and accurate response.
    - Ensure the response is grammatically correct and engaging.
    """
    response = gemini.generate_content(prompt)
    return response.text

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], 
                        avatar="🙋" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        try:
            # Find the closest answer
            retrieved_answer = find_closest_question(prompt, faiss_index, df)
            if retrieved_answer:
                # Generate a refined answer using Gemini
                refined_answer = generate_refined_answer(prompt, retrieved_answer)
                response = f"**Puji**:\n{refined_answer}"
            else:
                response = "**Puji**:\nI'm sorry, I cannot answer that question."
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()