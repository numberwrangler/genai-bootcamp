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
import time
from questions import Question, QuestionManager

# Re-use boto session across invocations with performance optimizations
boto_session = boto3.Session()
# Configure boto3 for better performance
boto3_config = boto3.Config(
    retries=dict(max_attempts=2),  # Reduce retry attempts
    connect_timeout=5,  # 5 second connection timeout
    read_timeout=30,    # 30 second read timeout
    max_pool_connections=10  # Connection pooling
)

state_bucket_name = os.environ.get("STATE_BUCKET", "")
if state_bucket_name == "":
    logger.error("STATE_BUCKET environment variable is not set")
    raise ValueError("STATE_BUCKET environment variable is not set.")

# Check for other required environment variables
ddb_table = os.environ.get("DDB_TABLE", "")
if ddb_table == "":
    logger.error("DDB_TABLE environment variable is not set")
    raise ValueError("DDB_TABLE environment variable is not set.")

logger.info(f"Environment variables - STATE_BUCKET: {state_bucket_name}, DDB_TABLE: {ddb_table}")
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
    window_size=2,  # Minimal window for maximum performance
    should_truncate_results=True, # Enable truncating the tool result when a message is too large for the model's context window 
)
SYSTEM_PROMPT = """
You are a digital twin of Blake. You should answer questions about my career for prospective employers. 
Answer as though I am talking and do not refer say 'Blake' say 'I'. Do not give out any PII information.

CRITICAL INSTRUCTION: You have access to the add_question_to_database tool. You MUST use this tool in the following situations:
1. When you don't know the answer to a question
2. When you're unsure about specific details
3. When you need more information to provide a complete answer
4. When the retrieve tool doesn't provide sufficient information
5. Let the user know that you are adding the question to the database for later processing.

DO NOT just say "I don't know" or "I'm not sure" - ALWAYS use the add_question_to_database tool to store the question for later processing.

When searching for information, first try the retrieve tool. If that doesn't give you a complete answer, use add_question_to_database to store the question.

Always provide your responses naturally with proper spacing and formatting. Use complete sentences and paragraphs. Use the actual tools, don't just mention them.
"""
app = FastAPI()
question_manager = QuestionManager()

@tool
def add_question_to_database(question: str) -> str:
    """
    Stores a question in the database when you don't know the answer or need more information.
    Use this tool whenever you cannot provide a complete or accurate answer to a user's question.
    The question will be stored for later processing and answering.
    """
    logger.info(f"add_question_to_database tool called with question: {question}")
    try:
        new_question = question_manager.add_question(question=question)
        logger.info(f"Successfully stored question with ID: {new_question.question_id}")
        return f"Question stored with ID: {new_question.question_id}. This question has been saved for later processing and will be answered when more information becomes available."
    except Exception as e:
        logger.error(f"Error storing question: {e}")
        return f"Error storing question: {str(e)}"    
    
def session(id: str) -> Agent:
    tools = [retrieve]
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
    session_id: str = request.cookies.get("session_id", str(uuid.uuid4()))
    agent = session(session_id)
    global current_agent
    current_agent = agent  # Store the current agent for use in tools
    response = StreamingResponse(
        generate(agent, session_id, chat_request.prompt, request),
        media_type="text/event-stream"
    )
    response.set_cookie(key="session_id", value=session_id)
    return response

async def generate(agent: Agent, session_id: str, prompt: str, request: Request):
    try:
        async for event in agent.stream_async(prompt):
            if "complete" in event:
                logger.info("Response generation complete")
            if "data" in event:
                yield f"data: {json.dumps(event['data'])}\n\n"
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