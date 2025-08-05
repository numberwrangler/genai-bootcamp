# Admin API
import logging
from fastapi import FastAPI, Depends, HTTPException, Header, status

from auth import CognitoValidator
from questions import Question, QuestionManager, AnswerRequest, AddQuestionRequest, UpdateQuestionRequest, SyncResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
cognito_validator = CognitoValidator()
question_manager = QuestionManager()


def get_current_user(authorization: str = Header(..., description="Bearer token for authentication")):
    """
    Dependency to validate Cognito JWT token from Authorization header.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme.")
    token = authorization.split(" ")[1]
    try:
        payload = cognito_validator.validate_token(token)
        return payload
    except ValueError as e:
        logger.error(f"Token validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


@app.get("/")
async def root():
    return {"message": "OK"}


@app.get("/adminapi/questions", response_model=list[Question])
async def list_questions(
    unansweredOnly: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """
    Lists all questions.

    An optional query parameter 'unansweredOnly' can be used to filter for
    unanswered questions.

    Requires authentication.
    """
    logger.info(f"Received request to list questions, unansweredOnly={unansweredOnly}")
    try:
        return question_manager.list_questions(unanswered_only=unansweredOnly)
    except Exception as e:
        logger.error(f"Error listing questions: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred while listing questions.")


@app.post("/adminapi/questions", response_model=Question, status_code=status.HTTP_201_CREATED)
async def add_question(
    add_request: AddQuestionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Adds a new question.

    Accepts a question with or without an answer. The processed flag will be
    set to False.

    Requires authentication.
    """
    return question_manager.add_question(
        question=add_request.question,
        answer=add_request.answer
    )


@app.post("/adminapi/questions/{question_id}/answer", response_model=Question)
async def answer_question(
    question_id: str,
    answer_request: AnswerRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Adds an answer to a question.

    Requires authentication.
    """
    try:
        return question_manager.answer_question(
            question_id=question_id,
            answer=answer_request.answer
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/adminapi/questions/{question_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_question(
    question_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Deletes a question.

    Requires authentication.
    """
    try:
        question_manager.delete_question(question_id=question_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/adminapi/questions/{question_id}/update", response_model=Question)
async def update_question(
    question_id: str,
    update_request: UpdateQuestionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Updates a question.

    The question, answer or both can be updated.
    After update, the processed flag is set to False.

    Requires authentication.
    """
    logger.info(f"Received request to update question {question_id} with data: {update_request.model_dump_json(exclude_none=True)}")
    if update_request.question is None and update_request.answer is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'question' or 'answer' must be provided."
        )
    try:
        return question_manager.update_question(
            question_id=question_id,
            question=update_request.question,
            answer=update_request.answer
        )
    except ValueError as e:
        logger.error(f"Could not update question {question_id}: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred updating question {question_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred while updating the question.")


@app.post("/adminapi/sync", response_model=SyncResponse, status_code=status.HTTP_202_ACCEPTED)
async def sync_knowledge_base(current_user: dict = Depends(get_current_user)):
    """
    Triggers a sync of the questions to the knowledge base.
    This is an asynchronous operation.

    Requires authentication.
    """
    logger.info("Received request to sync knowledge base")
    try:
        return question_manager.sync_to_knowledge_base()
    except Exception as e:
        logger.error(f"Error syncing knowledge base: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred while syncing the knowledge base.")

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
