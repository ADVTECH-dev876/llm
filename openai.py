# Install dependencies before running:
# pip install fastapi uvicorn openai python-dotenv

from fastapi import FastAPI, Request
import openai
import os
from dotenv import load_dotenv

load_dotenv()  # Loads your OpenAI key from .env
openai.api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI()

@app.post("/ask")
async def ask_llm(request: Request):
    req_json = await request.json()
    question = req_json.get("question", "")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or use "gpt-4"
        messages=[{"role": "user", "content": question}],
        max_tokens=400
    )
    answer = response["choices"][0]["message"]["content"]
    return {"answer": answer}

# Run with: uvicorn myapp:app --reload
