from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_verifier import RAGVerifier

# Initialize FastAPI app
api = FastAPI()

# Enable CORS (so frontend on Vercel can call backend)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI verifier
verifier = RAGVerifier()

# Request model
class VerificationRequest(BaseModel):
    question: str
    answer: str
    subject: str

# Root route
@api.get("/")
def read_root():
    return {"status": "AI Verifier API is running."}

# Verify route
@api.post("/verify")
def handle_verification(request: VerificationRequest):
    """Receives data from the frontend and returns the AI's evaluation."""
    result = verifier.verify_answer(request.question, request.answer)
    return result

# 👇 Required for Vercel to detect FastAPI app
app = api
