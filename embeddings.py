import json
import pickle
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("faq.json", "r", encoding="utf-8") as f:
    faq = json.load(f)

questions = [item["question"] for item in faq]

print("Generating embeddings...")
embeddings = model.encode(questions)

output_path = "vector_store.pkl" # Save vector store

with open(output_path, "wb") as f:
    pickle.dump((questions, embeddings, faq), f)

print(" Embeddings generated!")
print(f" Saved to: {os.path.abspath(output_path)}")
