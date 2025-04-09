import os
import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Load secrets
api_key = st.secrets["PINECONE_API_KEY"]
environment = st.secrets["PINECONE_ENV"]
index_name = st.secrets["PINECONE_INDEX"]
gemini_key = st.secrets["GEMINI_API_KEY"]

# Initialize Pinecone client
pc = Pinecone(api_key=api_key, environment=environment)
index = pc.Index(index_name)

# Load BERT-based model
model = SentenceTransformer('bert-base-uncased')

def generate_embeddings(text):
    return model.encode([text])[0]

def query_pinecone(query, top_k=3):
    query_embedding = generate_embeddings(query)
    query_results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    return [result['metadata']['text'] for result in query_results['matches']]

def generate_response(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def generate_response_with_context(relevant_chunks, user_query, api_key):
    detailed_prompt = (
        f"You are a helpful, knowledgeable assistant with expertise in HIV and AIDS. "
        f"Additionally, based on the user query, here are some relevant insights: {relevant_chunks}. "
        f"Provide an empathetic and helpful response to the following user query: {user_query}. "
    )
    return generate_response(detailed_prompt, api_key)

def main():
    st.title('HIV Information Chatbot')
    st.write("HIV and AIDS are manageable conditions. Ask anything below!")

    user_query = st.text_input("Ask a question about HIV or AIDS:")

    if user_query:
        relevant_chunks = query_pinecone(user_query)
        formatted_chunks = "\n".join(relevant_chunks)
        response = generate_response_with_context(formatted_chunks, user_query, gemini_key)

        if response:
            st.write(f"**Chatbot Response:** {response}")
        else:
            st.write("❌ Failed to generate a response.")

if __name__ == "__main__":
    main()
