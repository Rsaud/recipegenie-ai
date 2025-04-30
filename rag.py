# Convert Excel Data to a Textual Format
import pandas as pd

# Load Excel
df = pd.read_excel("final_food_rec_with_type.xlsx")


# Convert rows to documents
documents = []
for _, row in df.iterrows():
    doc = f"""البلد: {row['البلد']}
النوع: {row['النوع']}
الاسم: {row['الاسم']}
المكونات: {row['المكونات']}
الطريقة: {row['الطريقة']}"""
    documents.append(doc)


# Chunk and Embed the Data
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# Store Embeddings in a Vector Database
import faiss
import numpy as np
import google.generativeai as genai

index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

# Set Up the RAG Pipeline
query = "باذنجان ، بصل، ثوم، دجاج"
query_embedding = model.encode([query])
D, I = index.search(query_embedding, k=3)

# Get the top results
retrieved_docs = [documents[i] for i in I[0]]

# Concatenate for LLM
context = "\n\n".join(retrieved_docs)
print(context)
# prompt = f"{context}\n\nالسؤال: {query}\nالإجابة:"
prompt = f"""
    You are a creative recipe generator and understand many languages. Based on the following ingredients: {query},
    and inspired by these example recipes:

    {context}

    Create a recipe using the provided ingredients. Format it like a cooking blog and in the user input language.
    """


# Configure Gemini API
import os
os.environ["API_KEY"] = 'AIzaSyDie47yNvkl8VdC1xR48joUkb9gZ9Tag6M'
genai.configure(api_key=os.environ["API_KEY"])
model_ai = genai.GenerativeModel('gemini-1.5-flash-latest')
response = model_ai.generate_content(prompt)
print(response.text.strip())
# Use RAG to Answer Queries

