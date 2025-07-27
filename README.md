# Fullstack AI Agent App

## Backend

- Python (FastAPI, agno, groq)
- Package management: uv
- AI agents via agno, using Groq models

## Frontend

- Next.js (TypeScript)
- shadcn/ui for sleek, modern UI

## Connection

- FastAPI endpoints connect backend and frontend

## Setup

- Backend: `uv venv .venv && source .venv/bin/activate && uv pip install fastapi uvicorn agno groq`
- Frontend: `cd frontend && npm install`

## Run

- Backend: `uvicorn main:app --reload`
- Frontend: `cd frontend && npm run dev`

## Notes

- All commands are Linux/zsh compatible
- Use #context7 and #search for best practices
