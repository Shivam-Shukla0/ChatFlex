import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API key from environment variable
# Get free key from: https://aistudio.google.com/apikey
api_key = os.environ.get("GEMINI_API_KEY", "")

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None


def generateResponse(query, context):
    """Use Google Gemini API to generate a response."""
    if not model:
        # Fallback: return context directly if no API key set
        if context and context != "No relevant context found in the dataset for your question.":
            return f"Based on the dataset: {context[:500]}"
        return "No API key configured. Please set GEMINI_API_KEY in your .env file."
    try:
        prompt = (
            f"You are a helpful assistant. Answer the question based only on the context provided.\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"
