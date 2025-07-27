from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
class AgentRequest(BaseModel):
    agent: str  # e.g., "youtube", "articles", "linkedin"
    query: str

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

@app.get("/")
async def read_root():
    return {"message": "Hello World"}


@app.post("/agent/ask", response_model=AgentResponse)
async def ask_agent(request: AgentRequest):
    try:
        if request.agent == "youtube":
            run = youtube_agent.run(request.query)
            answer = run.content
        elif request.agent == "articles":
            run = articles_agent.run(request.query)
            answer = run.content
        elif request.agent == "linkedin":
            run = linkedin_agent.run(request.query)
            answer = run.content
        elif request.agent == "finance":
            run = finance_agent.run(request.query)
            answer = run.content
        elif request.agent == "web":
            run = web_agent.run(request.query)
            answer = run.content
        else:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": request.query}],
                model="llama3-8b-8192",
            )
            answer = chat_completion.choices[0].message.content
        return AgentResponse(response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
