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

When searching for information via a tool, use the tool to retrieve it, or if you don't know the answer, use the add_question_to_database tool.

Always provide your responses naturally with proper spacing and formatting. Use complete sentences and paragraphs. Do not display tool calls as text - use the actual tools.
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
    # Re-enable tools now that we know the agent works
    tools = [retrieve, add_question_to_database, type_out_text]
    logger.info(f"Available tools: {[tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in tools]}")
    
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
        # Clean up any empty messages and orphaned tool_use blocks
        original_count = len(agent.messages)
        
        # More robust cleanup to handle orphaned tool_use blocks
        cleaned_messages = []
        i = 0
        while i < len(agent.messages):
            msg = agent.messages[i]
            
            # Skip empty messages
            if not msg.get("content") or len(msg["content"]) == 0:
                i += 1
                continue
            
            # Check if this message has tool_use blocks
            has_tool_use = any(
                content.get("type") == "tool_use" 
                for content in msg.get("content", [])
            )
            
            if has_tool_use:
                # Check if the next message has corresponding tool_result
                if i + 1 < len(agent.messages):
                    next_msg = agent.messages[i + 1]
                    has_tool_result = any(
                        content.get("type") == "tool_result" 
                        for content in next_msg.get("content", [])
                    )
                    
                    if has_tool_result:
                        # Both tool_use and tool_result are present, keep both
                        cleaned_messages.append(msg)
                        cleaned_messages.append(next_msg)
                        i += 2  # Skip the next message since we already added it
                    else:
                        # Orphaned tool_use, skip this message
                        logger.warning(f"Skipping orphaned tool_use message at index {i}")
                        i += 1
                else:
                    # tool_use at the end without tool_result, skip it
                    logger.warning(f"Skipping orphaned tool_use message at end of conversation")
                    i += 1
            else:
                # Regular message, keep it
                cleaned_messages.append(msg)
                i += 1
        
        agent.messages = cleaned_messages
        cleaned_count = len(agent.messages)
        if original_count != cleaned_count:
            logger.info(f"Cleaned {original_count - cleaned_count} problematic messages from conversation history")
        
        # Log message count for debugging
        logger.info(f"Processing chat with {len(agent.messages)} messages in history")
        
        # If we still have too many messages or suspect issues, start fresh
        if len(agent.messages) > 10:
            logger.warning("Too many messages in history, starting fresh conversation")
            agent.messages = []
        
        full_response = ""
        event_count = 0
        timeout_count = 0
        max_timeout = 30  # 30 seconds timeout
        
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
                    logger.info(f"Received data: '{event['data']}'")
                    # Ensure proper formatting and handle special characters
                    data = event['data'].replace('\n', ' ').strip()
                    # Fix common spacing issues
                    data = data.replace('  ', ' ')  # Remove double spaces
                    data = data.replace(' ,', ',')  # Fix space before comma
                    data = data.replace(' .', '.')  # Fix space before period
                    data = data.replace(' ?', '?')  # Fix space before question mark
                    data = data.replace(' !', '!')  # Fix space before exclamation
                    if data:
                        yield f"data: {data}\n\n"
                else:
                    logger.warning(f"Unknown event type: {event}")
                
                # Add timeout protection
                await asyncio.sleep(0.1)  # Small delay to prevent blocking
                
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
                logger.error("Detected orphaned tool_use blocks, clearing conversation history")
                agent.messages = []
                yield f"data: [Error: Conversation corrupted. Please try your question again.]\n\n"
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


# Called by the Lambda Adapter to check liveness
@app.get("/")
async def root():
    return Response(
        content=json.dumps({"message": "OK"}),
        media_type="application/json",
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
