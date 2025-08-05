from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands import Agent, tool
from strands.models import BedrockModel
from strands.session.s3_session_manager import S3SessionManager
from strands_tools import retrieve
import boto3
import json
import logging
import os
import uuid
import uvicorn
import asyncio
from questions import Question, QuestionManager

# Re-use boto session across invocations
boto_session = boto3.Session()
state_bucket_name = os.environ.get("STATE_BUCKET", "")
if state_bucket_name == "":
    raise ValueError("BUCKET_NAME environment variable is not set.")
logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s", 
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
model_id = os.environ.get("MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
bedrock_model = BedrockModel(
    model_id=model_id,
    # Add Guardrails here
)
current_agent: Agent | None = None
conversation_manager = SlidingWindowConversationManager(
    window_size=5,  # Reduced maximum number of messages to keep
    should_truncate_results=True, # Enable truncating the tool result when a message is too large for the model's context window 
)
SYSTEM_PROMPT = """
You are a digital twin of Blake. You should answer questions about my career for prospective employers. Answer as though I am talking. Do not give out any PII information.

When searching for information via a tool, use the tool to retrieve it, or if you don't know the answer, use the tool add_question_to_database tool.
Return the question_id.

Always provide your responses naturally. The typewriter effect will be handled automatically.
"""
app = FastAPI()
question_manager = QuestionManager()

@tool
def add_question_to_database(question: str) -> str:
    """
    Adds a new unanswered question to DynamoDB for later processing.
    """
    new_question = question_manager.add_question(question=question)
    return f"Question stored with ID: {new_question.question_id}. Awaiting answer."

@tool
def type_out_text(answer: str) -> str:
    """
    Types out the answer character by character for a typewriter effect.
    This tool should be used for the final response to create a typewriter effect.
    """
    # Return the full answer - the streaming will handle the typewriter effect
    return answer
    
    
def session(id: str) -> Agent:
    tools = [retrieve, add_question_to_database, type_out_text]
    session_manager = S3SessionManager(
        boto_session=boto_session,
        bucket=state_bucket_name,
        session_id=id,
    )
    return Agent(
        conversation_manager=conversation_manager,
        model=bedrock_model,
        session_manager=session_manager,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
    )
    
class ChatRequest(BaseModel):
    prompt: str

@app.post('/api/chat')
async def chat(chat_request: ChatRequest, request: Request):
    # Validate that the prompt is not empty
    if not chat_request.prompt or not chat_request.prompt.strip():
        return Response(
            content=json.dumps({"error": "Prompt cannot be empty"}),
            media_type="application/json",
            status_code=400
        )
    
    session_id: str = request.cookies.get("session_id", str(uuid.uuid4()))
    agent = session(session_id)
    global current_agent
    current_agent = agent  # Store the current agent for use in tools
    response = StreamingResponse(
        generate(agent, session_id, chat_request.prompt.strip(), request),
        media_type="text/event-stream"
    )
    response.set_cookie(key="session_id", value=session_id)
    return response

async def generate(agent: Agent, session_id: str, prompt: str, request: Request):
    try:
        # Clean up any empty messages before processing
        original_count = len(agent.messages)
        agent.messages = [msg for msg in agent.messages if 
                         msg.get("content") and 
                         len(msg["content"]) > 0 and 
                         msg["content"][0].get("text", "").strip()]
        cleaned_count = len(agent.messages)
        if original_count != cleaned_count:
            logger.info(f"Cleaned {original_count - cleaned_count} empty messages from conversation history")
        
        # Log message count for debugging
        logger.info(f"Processing chat with {len(agent.messages)} messages in history")
        
        full_response = ""
        async for event in agent.stream_async(prompt):
            if "complete" in event:
                logger.info("Response generation complete")
            if "data" in event:
                full_response += event['data']
                # Create typewriter effect by yielding one character at a time
                for char in event['data']:
                    yield f"data: {char}\n\n"
                    # Add a small delay for typewriter effect
                    await asyncio.sleep(0.03)  # 30ms delay between characters
    except Exception as e:
        error_message = json.dumps({"error": str(e)})
        yield f"event: error\ndata: {error_message}\n\n"

@app.get('/api/chat')
def chat_get(request: Request):
    session_id = request.cookies.get("session_id", str(uuid.uuid4()))
    agent = session(session_id)

    # Filter messages to only include first text content
    filtered_messages = []
    for message in agent.messages:
        if (message.get("content") and 
            len(message["content"]) > 0 and 
            "text" in message["content"][0]):
            filtered_messages.append({
                "role": message["role"],
                "content": [{
                    "text": message["content"][0]["text"]
                }]
            })
 
    response = Response(
        content=json.dumps({
            "messages": filtered_messages,
        }),
        media_type="application/json",
    )
    response.set_cookie(key="session_id", value=session_id)
    return response


# Called by the Lambda Adapter to check liveness
@app.get("/")
async def root():
    return Response(
        content=json.dumps({"message": "OK"}),
        media_type="application/json",
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
