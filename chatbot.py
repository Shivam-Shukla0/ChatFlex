import google.generativeai as genai

# Configure your Gemini API key
genai.configure(api_key="AIzaSyAlp6HNNVByQhgNlkghrUsD0zp1lRppqvI")

# Initialize model
model = genai.GenerativeModel("gemini-2.5-pro")


def generateResponse(query, context):
    """Use Google Gemini API to generate a response."""
    try:
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {e}"
