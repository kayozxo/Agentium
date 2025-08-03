import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import io
import os
from dotenv import load_dotenv
from PIL import Image
from agents import get_agent
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Verify API key is loaded
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

logger.info(f"API Key loaded: {groq_api_key[:10]}..." if groq_api_key else "No API key found")

app = FastAPI(title="Agentium API", version="1.0.0")

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileData(BaseModel):
    type: str  # 'image' or 'file'
    name: str
    data: Optional[str] = None  # base64 encoded for images
    mimeType: Optional[str] = None
    size: Optional[int] = None

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str
    files: Optional[List[FileData]] = None

class ChatRequest(BaseModel):
    agent: str
    query: str
    messages: List[ChatMessage]
    useVisionModel: Optional[bool] = False
    files: Optional[List[FileData]] = None

class ChatResponse(BaseModel):
    response: str
    agent_used: str
    model_used: str

@app.post("/agent/ask", response_model=ChatResponse)
async def ask_agent(request: ChatRequest):
    try:
        logger.info(f"Request received: agent={request.agent}, useVisionModel={request.useVisionModel}")
        logger.info(f"Query: {request.query[:100]}...")  # Log first 100 chars of query

        # Validate request
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if not request.agent:
            raise HTTPException(status_code=400, detail="Agent type must be specified")

        # Get the appropriate agent
        try:
            agent = get_agent(request.agent, request.useVisionModel)
            if agent is None:
                raise HTTPException(status_code=400, detail=f"Invalid agent type: {request.agent}")
        except Exception as e:
            logger.error(f"Error getting agent: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {str(e)}")

        # Prepare the query
        query = request.query.strip()

        # For vision model with images, we need special handling
        if request.useVisionModel and request.files and any(f.type == "image" for f in request.files):
            logger.info("Processing vision model request with images")

            try:
                # For Agno vision models, we need to pass images as part of the message
                images_data = []

                for file_data in request.files:
                    if file_data.type == "image" and file_data.data:
                        # Remove data URL prefix if present (data:image/jpeg;base64,)
                        image_data = file_data.data
                        if image_data.startswith('data:'):
                            image_data = image_data.split(',')[1]

                        images_data.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file_data.mimeType};base64,{image_data}"
                            }
                        })

                if images_data:
                    logger.info(f"Processing {len(images_data)} images for vision model")

                    # For Agno, we can pass images directly in the run method
                    # The agent should handle multimodal input automatically
                    try:
                        # Method 1: Try passing images as a parameter to run()
                        run_response = agent.run(query, images=images_data)
                        response = run_response.content if hasattr(run_response, 'content') else str(run_response)
                        logger.info("Vision processing successful with images parameter")
                    except Exception as e1:
                        logger.warning(f"Method 1 failed: {e1}, trying method 2")
                        try:
                            # Method 2: Try creating a multimodal message structure
                            multimodal_message = {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": query}
                                ] + [
                                    {"type": "image_url", "image_url": img["image_url"]}
                                    for img in images_data
                                ]
                            }
                            run_response = agent.run(multimodal_message)
                            response = run_response.content if hasattr(run_response, 'content') else str(run_response)
                            logger.info("Vision processing successful with multimodal message")
                        except Exception as e2:
                            logger.warning(f"Method 2 failed: {e2}, trying method 3")
                            try:
                                # Method 3: Try using the image data directly with base64
                                image_prompt = f"{query}\n\n[Image data provided as base64]"
                                # Pass the first image as base64 directly
                                first_image = images_data[0]["image_url"]["url"]
                                run_response = agent.run(image_prompt, image=first_image)
                                response = run_response.content if hasattr(run_response, 'content') else str(run_response)
                                logger.info("Vision processing successful with direct base64")
                            except Exception as e3:
                                logger.error(f"All vision methods failed: {e1}, {e2}, {e3}")
                                # Fallback to text-only with image description
                                fallback_query = f"{query}\n\nNote: I received {len(images_data)} image(s) but couldn't process them visually. Please describe the image content if you need specific analysis."
                                run_response = agent.run(fallback_query)
                                response = run_response.content if hasattr(run_response, 'content') else str(run_response)
                                response = f"⚠️ **Vision Processing Issue**: {response}"
                else:
                    # No valid images found
                    run_response = agent.run(query)
                    response = run_response.content if hasattr(run_response, 'content') else str(run_response)

            except Exception as e:
                logger.error(f"Error processing vision request: {str(e)}")
                # Fallback to regular text processing
                run_response = agent.run(f"{query}\n\n[Note: Image processing failed - {str(e)}]")
                response = run_response.content if hasattr(run_response, 'content') else str(run_response)

        else:
            logger.info("Processing regular model request")

            # Build conversation context with error handling
            conversation_context = ""
            try:
                if request.messages and len(request.messages) > 1:
                    # Add recent conversation history for context
                    recent_messages = request.messages[-6:]  # Last 6 messages for context
                    for msg in recent_messages[:-1]:  # Exclude current message
                        if msg.content:  # Only add non-empty messages
                            role_label = "Human" if msg.role == "user" else "Assistant"
                            conversation_context += f"{role_label}: {msg.content[:500]}\n"  # Limit message length
            except Exception as e:
                logger.warning(f"Error building conversation context: {str(e)}")
                conversation_context = ""

            # Combine context with current query
            full_query = f"{conversation_context}\nHuman: {query}" if conversation_context else query

            # Limit total query length to prevent issues
            if len(full_query) > 4000:
                full_query = full_query[-4000:]  # Keep last 4000 characters
                logger.warning("Query truncated due to length")

            # Call the agent with error handling
            try:
                logger.info("Calling agent.run()...")
                run_response = agent.run(full_query)
                response = run_response.content if hasattr(run_response, 'content') else str(run_response)
                logger.info("Agent response received successfully")
            except Exception as e:
                logger.error(f"Error running agent: {str(e)}")
                # Try with just the original query if context caused issues
                try:
                    logger.info("Retrying with simple query...")
                    run_response = agent.run(query)
                    response = run_response.content if hasattr(run_response, 'content') else str(run_response)
                    logger.info("Retry successful")
                except Exception as retry_error:
                    logger.error(f"Retry also failed: {str(retry_error)}")
                    raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(retry_error)}")

        # Validate response
        if not response:
            logger.warning("Empty response from agent")
            response = "I apologize, but I couldn't generate a response. Please try again."

        logger.info("Response generated successfully")

        # Get model ID safely
        try:
            model_id = agent.model.id if hasattr(agent.model, 'id') else str(agent.model)
        except:
            model_id = "unknown"

        return ChatResponse(
            response=str(response),  # Ensure response is string
            agent_used=request.agent,
            model_used=model_id
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ask_agent: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Agentium API is running"}

@app.get("/health")
async def health_check():
    try:
        # Test basic functionality
        groq_key = os.getenv("GROQ_API_KEY")
        return {
            "status": "healthy",
            "message": "API is working correctly",
            "has_groq_key": bool(groq_key),
            "groq_key_prefix": groq_key[:10] if groq_key else None
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/agents")
async def list_agents():
    """List available agents"""
    try:
        return {
            "agents": [
                {"id": "general", "name": "General Chat", "description": "General purpose assistant"},
                {"id": "web", "name": "Web Search", "description": "Web research and search"},
                {"id": "youtube", "name": "YouTube", "description": "YouTube content analysis"},
                {"id": "articles", "name": "Articles", "description": "Article research and analysis"},
                {"id": "linkedin", "name": "LinkedIn", "description": "LinkedIn content generation"},
                {"id": "finance", "name": "Finance", "description": "Financial analysis and data"}
            ]
        }
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

# Add startup event to test agent initialization
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Testing agent initialization...")
        test_agent = get_agent("general", False)
        logger.info(f"General agent initialized successfully: {test_agent}")
    except Exception as e:
        logger.error(f"Failed to initialize agents on startup: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)