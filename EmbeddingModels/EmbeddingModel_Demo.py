from langchain_google_genai import GoogleGenerativeAI, GeminiEmbeddings
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
load_dotenv()
embedding = GeminiEmbeddings(model='gemini-1.5-pro', temperature=0, max_completion_tokens=10, openai_api_key=os.getenv("GOOGLE_API_KEY"))
# model = GoogleGenerativeAI(model='gemini-1.5-pro', temperature=0, max_completion_tokens=10, openai_api_key=os.getenv("GOOGLE_API_KEY"))
documentts = [
    "What is the capital of France?",
    "What is the capital of Spain?",
    "What is the capital of Italy?",
    "What is the capital of Germany?",
    "What is the capital of Canada?",
]

query = "What is the capital of France?"
query_embedding = embedding.embed_query(query)
document_embeddings = embedding.embed_documents(documentts)
# document_embeddings = embedding.embed_documents(documentts)
scores = cosine_similarity([query_embedding], document_embeddings)[0]
index,score = sort(list(enumerate(scores)), key=lambda x:x[1])[-1]
print(documents[index])