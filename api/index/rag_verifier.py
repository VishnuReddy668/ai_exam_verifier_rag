import os
import re
import json
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class RAGVerifier:
    def __init__(self):
        # Configure the Groq API client
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Initialize the components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db_client = chromadb.Client()
        self.collection = self.db_client.get_or_create_collection(name="exam_knowledge")
        self._load_and_index_documents()

    def _load_and_index_documents(self):
        """Loads documents from the knowledge_base and indexes them in ChromaDB."""
        knowledge_path = 'knowledge_base'
        if not os.path.exists(knowledge_path):
            print("Knowledge base directory not found.")
            return

        documents, ids = [], []
        for i, filename in enumerate(os.listdir(knowledge_path)):
            if filename.endswith(".txt"):
                with open(os.path.join(knowledge_path, filename), 'r', encoding='utf-8') as f:
                    documents.append(f.read())
                    ids.append(f"doc_{i}")
        
        if documents:
            embeddings = self.embedding_model.encode(documents).tolist()
            self.collection.add(embeddings=embeddings, documents=documents, ids=ids)
            print(f"Indexed {len(documents)} documents.")

    def _parse_response(self, text_response: str) -> dict:
        """
        Parses the structured text response from the LLM into a JSON object.
        This version is more robust and handles missing fields.
        """
        data = {}
        
        # Helper function to safely find patterns
        def safe_search(pattern, text, default="N/A"):
            match = re.search(pattern, text)
            return match.group(1).strip() if match else default

        try:
            # Safely parse each field
            data['overall_score'] = int(safe_search(r"Overall Score: (\d+)", text_response, default="0"))
            
            data['accuracy'] = {
                "rating": safe_search(r"Accuracy: (.*?)\n", text_response),
                "feedback": safe_search(r"Accuracy Feedback: (.*?)\n", text_response)
            }
            data['completeness'] = {
                "rating": safe_search(r"Completeness: (.*?)\n", text_response),
                "feedback": safe_search(r"Completeness Feedback: (.*?)\n", text_response)
            }
            data['structure'] = {
                "rating": safe_search(r"Structure: (.*?)\n", text_response),
                "feedback": safe_search(r"Structure Feedback: (.*?)\n", text_response)
            }
            data['ai_feedback'] = {
                "strengths": safe_search(r"Strengths: (.*?)\n", text_response),
                "suggestions": safe_search(r"Suggestions: (.*)", text_response, re.DOTALL)
            }
            data['retrieved_sources'] = ["Retrieved from Vector DB"]
            
            return data
            
        except Exception as e:
            print(f"Critical error during parsing: {e}")
            return {"error": "Failed to parse LLM response", "details": text_response}

    # UPDATED FUNCTION
    def verify_answer(self, question: str, student_answer: str) -> dict:
        """Verifies the student's answer using RAG with Groq."""
        question_embedding = self.embedding_model.encode(question).tolist()
        retrieved_docs = self.collection.query(query_embeddings=[question_embedding], n_results=2)
        context = "\n".join(retrieved_docs['documents'][0])

        prompt = f"""
        You are an AI Exam Verifier. Evaluate the student's answer based only on the provided context.

        **Textbook Context:**
        {context}

        **Exam Question:**
        {question}

        **Student's Answer:**
        {student_answer}

        **Instructions:**
        Provide a detailed analysis in the following format. Do not add any other text or explanations.

        Overall Score: [score out of 100]
        Accuracy: [Excellent/Good/Needs Improvement]
        Accuracy Feedback: [Your brief feedback on accuracy]
        Completeness: [Excellent/Good/Needs Improvement]
        Completeness Feedback: [Your brief feedback on completeness]
        Structure: [Excellent/Good/Needs Improvement]
        Structure Feedback: [Your brief feedback on structure]
        Strengths: [A sentence on what the student did well]
        Suggestions: [A sentence on how to improve]
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            
            # Check if the response was blocked by safety filters or is otherwise empty
            if not chat_completion.choices:
                print("API response was blocked or empty.")
                return {"error": "The AI's response was blocked for safety reasons. Please try a different query."}

            response_text = chat_completion.choices[0].message.content
            return self._parse_response(response_text)

        except Exception as e:
            print(f"An error occurred with the API call: {e}")
            return {"error": "An unexpected error occurred while contacting the AI model."}