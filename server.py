import os
import pickle
import time
import uuid
from typing import Optional, List

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load env
# -------------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.2-3b-instruct:free"
)

# -------------------------------
# Load model + vector store
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("vector_store.pkl", "rb") as f:
    questions, embeddings, faq = pickle.load(f)

embeddings = np.array(embeddings)

# -------------------------------
# App
# -------------------------------
app = FastAPI(title="FAQ Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# In-memory sessions
# -------------------------------
SESSIONS = {}
# session_id -> {
#   last_user_question,
#   last_generated_answer,
#   last_faq_matches
# }

# -------------------------------
# Schemas
# -------------------------------
class AskRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    top_k: int = 3
    from_faq: bool = False


class MatchItem(BaseModel):
    question: str
    answer: str
    score: float

class AskResponse(BaseModel):
    session_id: str
    top_matches: List[MatchItem]
    generated_answer: str
    is_follow_up: bool

# -------------------------------
# LLM call
# -------------------------------
def query_llm(prompt: str, max_tokens=250, retries=3, wait=5):
    if not OPENROUTER_API_KEY:
        return None

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    for _ in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            return text if text else None
        except requests.exceptions.HTTPError:
            if r.status_code == 429:
                time.sleep(wait)
            else:
                return None
        except Exception:
            return None

    return None

# -------------------------------
# Retrieval
# -------------------------------
def retrieve_faq_matches(question: str, top_k: int):
    q_emb = model.encode([question])
    scores = cosine_similarity(q_emb, embeddings)[0]
    idxs = scores.argsort()[-top_k:][::-1]

    matches = [
        {
            "question": faq[i]["question"],
            "answer": faq[i]["answer"],
            "score": float(scores[i]),
        }
        for i in idxs
    ]

    max_score = matches[0]["score"] if matches else 0.0
    return matches, max_score

# -------------------------------
# Follow-up detection
# -------------------------------
FOLLOWUP_KEYWORDS = [
    "what about", "how about", "and", "also",
    "that", "it", "then", "why", "how long", "more"
]

def is_follow_up(new_q: str, old_q: Optional[str]) -> bool:
    if not old_q:
        return False

    short = len(new_q.split()) <= 4 and "?" in new_q

    keyword_hit = any(k in new_q.lower() for k in FOLLOWUP_KEYWORDS)

    sim = cosine_similarity(
        model.encode([new_q]),
        model.encode([old_q])
    )[0][0]

    return short or keyword_hit or sim > 0.55
def safe_answer(text: Optional[str], fallback: str) -> str:
    return text if isinstance(text, str) and text.strip() else fallback

# -------------------------------
# FAQ relevance threshold
# -------------------------------
FAQ_RELEVANCE_THRESHOLD = 0.45

# -------------------------------
# Core pipeline (ONE FLOW)
# -------------------------------
def handle_question(question: str, session_id: str, top_k: int, from_faq: bool):
    session = SESSIONS.get(session_id, {
        "last_user_question": None,
        "last_generated_answer": None,
        "last_faq_matches": None
    })
    # FAQ CLICK → elaborate ONLY the selected FAQ
    if from_faq and session.get("last_faq_matches"):
        clicked = question.strip().lower()

        matched_faq = next(
            (m for m in session["last_faq_matches"]
            if m["question"].strip().lower() == clicked),
            None
        )

        base_answer = (
            matched_faq["answer"]
            if matched_faq
            else session["last_faq_matches"][0]["answer"]
        )
        
        prompt = (
            "You are an HR FAQ assistant.\n\n"
            f"FAQ question:\n{question}\n\n"
            f"FAQ answer:\n{base_answer}\n\n"
            "Task: Provide a more detailed, clear, and user-friendly explanation of this FAQ. "
            "Do not introduce new topics or other FAQs. Stay strictly within this question."
        )

        answer = safe_answer(query_llm(prompt), base_answer)


        session["last_user_question"] = question
        session["last_generated_answer"] = answer
        SESSIONS[session_id] = session

        return session_id, [], answer, False


    # FOLLOW-UP
    if is_follow_up(question, session["last_user_question"]):
        prompt = (
            "This is a follow-up question.\n\n"
            f"Previous answer:\n{session['last_generated_answer']}\n\n"
            f"User follow-up:\n{question}\n\n"
            "Based on these, provide a clear, user-friendly, slightly more detailed explanation, "
            "but keep it concise and easy to read."
        )

        answer = safe_answer(query_llm(prompt), session["last_generated_answer"] or "")

        session["last_user_question"] = question
        session["last_generated_answer"] = answer
        SESSIONS[session_id] = session

        return session_id, session["last_faq_matches"], answer, True

    # NORMAL QUESTION
    matches, max_score = retrieve_faq_matches(question, top_k)

   

    context = ""
    for m in matches:
        context += f"Q: {m['question']}\nA: {m['answer']}\n\n"
    prompt = (
        "You are an HR FAQ assistant.\n\n"
        f"User question:\n{question}\n\n"
        f"Relevant FAQ answers:\n{context}"
        "Task: Using the answers of the 3 provided FAQs, generate a single, clear, and user-friendly response "
        "that directly answers the user's question. You should combine and rephrase the information from the FAQs, "
        "but ensure the response is concise, accurate, and easy to understand."
    )

    fallback = matches[0]["answer"] if matches else "I'm sorry, I couldn't find an answer to that."
    answer = safe_answer(query_llm(prompt), fallback)


    session["last_user_question"] = question
    session["last_generated_answer"] = answer
    session["last_faq_matches"] = matches
    SESSIONS[session_id] = session

    return session_id, matches, answer, False

# -------------------------------
# Single API endpoint
# -------------------------------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())

    sid, matches, answer, follow = handle_question(
    req.question.strip(),
    session_id,
    req.top_k,
    req.from_faq
)


    return AskResponse(
        session_id=sid,
        top_matches=matches,
        generated_answer=answer,
        is_follow_up=follow
    )
