from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

from agno.agent import Agent
from groq import Groq
from agents import web_agent, finance_agent, youtube_agent, articles_agent, linkedin_agent

# --- FastAPI app setup ---
app = FastAPI()

# --- CORS for frontend dev ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic models ---
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class AgentRequest(BaseModel):
    agent: str
    query: str
    messages: Optional[List[Message]] = []  # Add conversation history

class AgentResponse(BaseModel):
    response: str

# --- Load .env and get Groq API key ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env file.")

# --- Agent setup ---
client = Groq(api_key=GROQ_API_KEY)
agent = Agent(model=client, markdown=True)

def create_contextual_query(messages: List[Message], current_query: str) -> str:
    """Create a contextual query including conversation history"""
    if not messages or len(messages) <= 1:
        return current_query

    # Get last few messages for context (excluding the current one)
    recent_messages = messages[-6:-1] if len(messages) > 1 else []

    if not recent_messages:
        return current_query

    context = "\n".join([
        f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
        for msg in recent_messages
    ])

    return f"Previous conversation:\n{context}\n\nCurrent question: {current_query}"

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.post("/agent/ask", response_model=AgentResponse)
async def ask_agent(request: AgentRequest):
    try:
        # Create contextual query with conversation history
        contextual_query = create_contextual_query(request.messages, request.query)

        if request.agent == "youtube":
            run = youtube_agent.run(contextual_query)
            answer = run.content
        elif request.agent == "articles":
            run = articles_agent.run(contextual_query)
            answer = run.content
        elif request.agent == "linkedin":
            run = linkedin_agent.run(contextual_query)
            answer = run.content
        elif request.agent == "finance":
            run = finance_agent.run(contextual_query)
            answer = run.content
        elif request.agent == "web":
            run = web_agent.run(contextual_query)
            answer = run.content
        else:
            # For general agent, use Groq messages format directly
            groq_messages = []
            for msg in request.messages[-10:]:  # Last 10 messages
                groq_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            # Add current query
            groq_messages.append({
                "role": "user",
                "content": request.query
            })

            chat_completion = client.chat.completions.create(
                messages=groq_messages,
                model="llama3-8b-8192",
            )
            answer = chat_completion.choices[0].message.content

        return AgentResponse(response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))