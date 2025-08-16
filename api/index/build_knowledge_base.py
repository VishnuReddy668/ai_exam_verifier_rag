import os
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Load your API key from the .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# --- Define the topics for all subjects ---

CS_TOPICS = [
    "Data Structures (Arrays, Linked Lists, Stacks, Queues)",
    "Algorithms (Sorting, Searching, Big O Notation)",
    "Object-Oriented Programming (Encapsulation, Inheritance, Polymorphism)",
    "Computer Networks (OSI Model, TCP/IP, HTTP)",
    "Operating Systems (Processes, Threads, Memory Management)",
    "Databases (SQL vs NoSQL, Normalization)",
    "Cybersecurity (Encryption, Firewalls, Malware)",
]

MATH_TOPICS = [
    "Calculus (Derivatives and Integrals)",
    "Linear Algebra (Vectors, Matrices, and Determinants)",
    "Probability and Statistics (Mean, Median, Standard Deviation)",
    "Differential Equations",
    "Trigonometry (Sine, Cosine, Tangent)",
]

PHYSICS_TOPICS = [
    "Newtonian Mechanics (Laws of Motion and Gravity)",
    "Thermodynamics (Laws and Principles)",
    "Electromagnetism (Coulomb's Law, Maxwell's Equations)",
    "Quantum Mechanics (Wave-particle duality, Schrodinger equation)",
    "Theory of Relativity (Special and General)",
]

# Combine all topics into a single list
ALL_TOPICS = CS_TOPICS + MATH_TOPICS + PHYSICS_TOPICS

def generate_and_save_knowledge(topics):
    """
    Generates a detailed explanation for each topic using an AI model
    and saves it to a corresponding .txt file.
    """
    output_dir = "knowledge_base"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for topic in topics:
        print(f"Generating knowledge for: {topic}...")
        
        # Create a clean filename
        filename = topic.split('(')[0].strip().lower().replace(' ', '_') + ".txt"
        filepath = os.path.join(output_dir, filename)

        # Create a prompt to generate the knowledge
        prompt = f"""
        Act as a university professor. Write a detailed, clear, and comprehensive explanation 
        of the following topic: '{topic}'.

        The explanation should be suitable for an undergraduate student.
        Start with a clear definition, explain the core concepts, and provide simple examples where possible.
        The content should be factual and well-structured.
        """
        
        try:
            # Generate content using the Gemini API
            response = model.generate_content(prompt)
            
            # Save the generated text to a file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Title: {topic}\n\n")
                f.write(response.text)
            
            print(f"Successfully saved knowledge to {filename}")

        except Exception as e:
            print(f"Could not generate knowledge for {topic}. Error: {e}")
        
        # Add a delay to avoid hitting API rate limits
        time.sleep(5) # Wait for 5 seconds between requests

if __name__ == "__main__":
    generate_and_save_knowledge(ALL_TOPICS)
    print("\nKnowledge base generation complete.")