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

# Re-use boto session across invocations
boto_session = boto3.Session()
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
    window_size=3,  # Further reduced to improve performance
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

@tool
def type_out_text(answer: str) -> str:
    """
    Types out the answer character by character for a typewriter effect.
    This tool should be used for the final response to create a typewriter effect.
    """
    # Return the full answer - the streaming will handle the typewriter effect
    return answer
    
    
def session(id: str, force_new: bool = False) -> Agent:
    # Start with just the essential tools to avoid tool_use issues
    tools = [retrieve, add_question_to_database]
    logger.info(f"Available tools: {[tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in tools]}")
    
    # If forcing new session, add timestamp to make it unique
    session_id_to_use = f"{id}_{int(time.time())}" if force_new else id
    
    session_manager = S3SessionManager(
        boto_session=boto_session,
        bucket=state_bucket_name,
        session_id=session_id_to_use,
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
    try:
        logger.info(f"Received chat request: {chat_request.prompt[:100]}...")
        
        # Validate that the prompt is not empty
        if not chat_request.prompt or not chat_request.prompt.strip():
            logger.warning("Empty prompt received")
            return Response(
                content=json.dumps({"error": "Prompt cannot be empty"}),
                media_type="application/json",
                status_code=400
            )
        
        session_id: str = request.cookies.get("session_id", str(uuid.uuid4()))
        logger.info(f"Using session ID: {session_id}")
        
        agent = session(session_id)
        global current_agent
        current_agent = agent  # Store the current agent for use in tools
        
        logger.info("Created agent successfully, starting streaming response")
        response = StreamingResponse(
            generate(agent, session_id, chat_request.prompt.strip(), request),
            media_type="text/event-stream"
        )
        response.set_cookie(key="session_id", value=session_id)
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return Response(
            content=json.dumps({"error": f"Internal server error: {str(e)}"}),
            media_type="application/json",
            status_code=500
        )

async def generate(agent: Agent, session_id: str, prompt: str, request: Request):
    try:
        # Check for any tool_use blocks in the conversation history
        has_tool_use = any(
            any(content.get("type") == "tool_use" for content in msg.get("content", []))
            for msg in agent.messages
        )
        
        if has_tool_use:
            logger.warning("Found tool_use blocks in conversation history, clearing to prevent validation errors")
            agent.messages = []
            # Also try to clear the S3 session
            try:
                if hasattr(agent, 'session_manager') and agent.session_manager:
                    agent.session_manager.clear_session()
                    logger.info("Cleared S3 session data due to tool_use blocks")
            except Exception as s3_error:
                logger.error(f"Error clearing S3 session: {s3_error}")
        
        # Clean up any empty messages
        original_count = len(agent.messages)
        agent.messages = [msg for msg in agent.messages if 
                         msg.get("content") and 
                         len(msg["content"]) > 0]
        
        cleaned_count = len(agent.messages)
        if original_count != cleaned_count:
            logger.info(f"Cleaned {original_count - cleaned_count} empty messages from conversation history")
        
        # Log message count for debugging
        logger.info(f"Processing chat with {len(agent.messages)} messages in history")
        
        # If we still have too many messages, start fresh
        if len(agent.messages) > 6:
            logger.warning("Too many messages in history, starting fresh conversation")
            agent.messages = []
        
        full_response = ""
        event_count = 0
        timeout_count = 0
        max_timeout = 15  # Reduced to 15 seconds timeout
        
        try:
            async for event in agent.stream_async(prompt):
                event_count += 1
                timeout_count = 0  # Reset timeout counter on any event
                logger.info(f"Stream event #{event_count}: {event}")  # Debug logging
                
                if "complete" in event:
                    logger.info("Response generation complete")
                    break
                elif "tool_use" in event:
                    # Handle tool calls - show that a tool is being used
                    tool_name = event.get("tool_use", {}).get("name", "unknown tool")
                    logger.info(f"Tool being used: {tool_name}")
                    yield f"data: [Using {tool_name}...]\n\n"
                elif "tool_result" in event:
                    # Handle tool results - show the result
                    tool_result = event.get("tool_result", {}).get("content", "No result")
                    logger.info(f"Tool result received: {tool_result[:100]}...")
                    yield f"data: [Tool result: {tool_result}]\n\n"
                elif "data" in event:
                    full_response += event['data']
                    data = event['data']
                    logger.info(f"Received data: '{data}'")
                    
                    if data:
                        # Send the data exactly as received, no buffering
                        yield f"data: {data}\n\n"
                else:
                    logger.warning(f"Unknown event type: {event}")
                
                # Add timeout protection - reduced delay
                await asyncio.sleep(0.01)  # Minimal delay to prevent blocking
                
            logger.info(f"Total events processed: {event_count}")
            if event_count == 0:
                logger.error("No events received from agent.stream_async!")
                yield f"data: [Error: No response generated]\n\n"
                
        except asyncio.TimeoutError:
            logger.error("Streaming timeout - agent took too long to respond")
            yield f"data: [Error: Response timeout]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            
            # Check if this is the specific tool_use validation error
            if "tool_use" in str(e) and "tool_result" in str(e):
                logger.error("Detected orphaned tool_use blocks, creating fresh session")
                try:
                    # Create a completely new session with a unique ID
                    new_session_id = f"{session_id}_fresh_{int(time.time())}"
                    agent = session(new_session_id, force_new=True)
                    global current_agent
                    current_agent = agent
                    logger.info(f"Created fresh session: {new_session_id}")
                except Exception as cleanup_error:
                    logger.error(f"Error creating fresh session: {cleanup_error}")
                
                yield f"data: [Error: Conversation corrupted. Starting fresh session. Please try your question again.]\n\n"
                return
            
            # Try fallback to non-streaming approach
            try:
                logger.info("Attempting fallback to non-streaming response")
                response = agent(prompt)
                logger.info(f"Fallback response: {response}")
                yield f"data: {str(response)}\n\n"
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                yield f"data: [Error: {str(e)}]\n\n"
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

@app.post('/api/clear')
def clear_chat(request: Request):
    """Clear the conversation history for the current session"""
    session_id = request.cookies.get("session_id", str(uuid.uuid4()))
    agent = session(session_id)
    agent.messages = []
    
    # Also clear the global agent to force a fresh session
    global current_agent
    current_agent = None
    
    response = Response(
        content=json.dumps({"message": "Conversation history cleared"}),
        media_type="application/json",
    )
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.get('/api/test-tools')
def test_tools(request: Request):
    """Test if tools are working properly"""
    try:
        # Test the retrieve tool
        test_result = retrieve("test query")
        logger.info(f"Retrieve tool test result: {test_result}")
        
        # Test the add_question_to_database tool
        test_question = add_question_to_database("test question")
        logger.info(f"Add question tool test result: {test_question}")
        
        return Response(
            content=json.dumps({
                "message": "Tools tested successfully",
                "retrieve_result": str(test_result),
                "add_question_result": str(test_question)
            }),
            media_type="application/json",
        )
    except Exception as e:
        logger.error(f"Tool test failed: {e}")
        return Response(
            content=json.dumps({"error": f"Tool test failed: {str(e)}"}),
            media_type="application/json",
            status_code=500
        )

@app.get('/api/test-agent')
def test_agent(request: Request):
    """Test if the agent can respond without tools"""
    try:
        session_id = str(uuid.uuid4())
        agent = session(session_id)
        
        # Test a simple response without tools
        test_prompt = "Hello, can you introduce yourself?"
        logger.info(f"Testing agent with prompt: {test_prompt}")
        
        # Use a simple call instead of streaming for testing
        response = agent(test_prompt)
        logger.info(f"Agent test response: {response}")
        
        return Response(
            content=json.dumps({
                "message": "Agent test successful",
                "response": str(response)
            }),
            media_type="application/json",
        )
    except Exception as e:
        logger.error(f"Agent test failed: {e}")
        return Response(
            content=json.dumps({"error": f"Agent test failed: {str(e)}"}),
            media_type="application/json",
            status_code=500
        )

@app.get('/api/test-unknown-question')
def test_unknown_question(request: Request):
    """Test if the agent uses add_question_to_database for unknown questions"""
    try:
        session_id = str(uuid.uuid4())
        agent = session(session_id)
        
        # Test with a question the agent shouldn't know
        test_prompt = "What was my exact salary at my first job in 2010?"
        logger.info(f"Testing agent with unknown question: {test_prompt}")
        
        # Use a simple call instead of streaming for testing
        response = agent(test_prompt)
        logger.info(f"Agent response to unknown question: {response}")
        
        return Response(
            content=json.dumps({
                "message": "Unknown question test completed",
                "prompt": test_prompt,
                "response": str(response)
            }),
            media_type="application/json",
        )
    except Exception as e:
        logger.error(f"Unknown question test failed: {e}")
        return Response(
            content=json.dumps({"error": f"Unknown question test failed: {str(e)}"}),
            media_type="application/json",
            status_code=500
        )

@app.get('/api/test-question-db')
def test_question_database(request: Request):
    """Test the question database functionality specifically"""
    try:
        # Test adding a question directly
        test_question = question_manager.add_question("Test question for database verification")
        logger.info(f"Question added successfully: {test_question.question_id}")
        
        return Response(
            content=json.dumps({
                "message": "Question database test successful",
                "question_id": test_question.question_id,
                "question": test_question.question,
                "processed": test_question.processed
            }),
            media_type="application/json",
        )
    except Exception as e:
        logger.error(f"Question database test failed: {e}")
        return Response(
            content=json.dumps({"error": f"Question database test failed: {str(e)}"}),
            media_type="application/json",
            status_code=500
        )


# Called by the Lambda Adapter to check liveness
@app.get("/")
async def root():
    return Response(
        content=json.dumps({
            "message": "OK", 
            "status": "healthy",
            "timestamp": time.time(),
            "environment": {
                "state_bucket": state_bucket_name,
                "ddb_table": os.environ.get("DDB_TABLE", "not_set"),
                "model_id": model_id
            }
        }),
        media_type="application/json",
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test basic functionality
        test_agent = session("health-check")
        return Response(
            content=json.dumps({
                "status": "healthy",
                "message": "Backend is running",
                "model_id": model_id,
                "bucket": state_bucket_name,
                "conversation_window": 3,
                "timeout": 15
            }),
            media_type="application/json",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response(
            content=json.dumps({
                "status": "unhealthy",
                "error": str(e)
            }),
            media_type="application/json",
            status_code=500
        )

@app.get("/api/performance-test")
async def performance_test():
    """Test response time"""
    start_time = time.time()
    try:
        session_id = str(uuid.uuid4())
        agent = session(session_id)
        
        # Simple test prompt
        test_prompt = "Hello"
        response = agent(test_prompt)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return Response(
            content=json.dumps({
                "status": "success",
                "response_time_seconds": round(response_time, 2),
                "response_length": len(str(response))
            }),
            media_type="application/json",
        )
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        logger.error(f"Performance test failed: {e}")
        return Response(
            content=json.dumps({
                "status": "error",
                "error": str(e),
                "response_time_seconds": round(response_time, 2)
            }),
            media_type="application/json",
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
