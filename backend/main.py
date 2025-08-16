from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_verifier import RAGVerifier

app = FastAPI()

# Allow requests from your frontend (important for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# Initialize our AI verifier
verifier = RAGVerifier()

# Define the structure of the request data
class VerificationRequest(BaseModel):
    question: str
    answer: str
    subject: str

@app.get("/")
def read_root():
    return {"status": "AI Verifier API is running."}

@app.post("/verify")
def handle_verification(request: VerificationRequest):
    """Receives data from the frontend and returns the AI's evaluation."""
    result = verifier.verify_answer(request.question, request.answer)
    return result