from google import genai
import streamlit as st 

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def explain_disease(label,confidence):
    prompt = f"""
    A plant disease classification model detected:

    Disease: {label}
    Confidence: {confidence:.2%}

    Explain this result in a simple and clear way.

    Include:
    1. What the disease is
    2. Common causes
    3. Common symptoms
    4. Recommended management
    5. Prevention

    Do not change the predicted disease.
    Make it clear that the classification is an AI prediction.
    """

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )
    return response.text