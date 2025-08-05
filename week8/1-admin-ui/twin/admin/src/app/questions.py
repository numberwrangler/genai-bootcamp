import uuid
import os
import boto3
import logging
from pydantic import BaseModel, Field
from botocore.exceptions import ClientError
from datetime import datetime


logger = logging.getLogger(__name__)


class Question(BaseModel):
    """
    Represents a question to be stored and managed.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    answer: str | None = None
    processed: bool = False


class AnswerRequest(BaseModel):
    """
    Represents the request body for answering a question.
    """
    answer: str


class AddQuestionRequest(BaseModel):
    """
    Represents the request body for adding a question.
    """
    question: str
    answer: str | None = None


class UpdateQuestionRequest(BaseModel):
    """
    Represents the request body for updating a question.
    """
    question: str | None = None
    answer: str | None = None


class IngestionJob(BaseModel):
    knowledgeBaseId: str
    dataSourceId: str
    ingestionJobId: str
    status: str
    createdAt: datetime = Field(alias='startedAt')
    updatedAt: datetime


class SyncResponse(BaseModel):
    status: str
    ingestionJob: IngestionJob | None = None


class QuestionManager:
    """
    Manages questions in DynamoDB.
    """
    def __init__(self):
        """
        Initialises the QuestionManager.
        - Sets up DynamoDB client.
        - Gets table name from environment variables.
        """
        logger.info("Initialising QuestionManager")
        self.dynamodb = boto3.resource('dynamodb')
        self.s3_client = boto3.client('s3')
        self.bedrock_agent_client = boto3.client('bedrock-agent')

        self.table_name = os.environ.get('DDB_TABLE')
        if not self.table_name:
            logger.error("DDB_TABLE environment variable is not set")
            raise ValueError("DDB_TABLE environment variable is required.")
        self.table = self.dynamodb.Table(self.table_name)
        logger.info(f"DynamoDB table name: {self.table_name}")

        self.kb_bucket = os.environ.get('KB_INPUT_BUCKET')
        self.kb_id = os.environ.get('KB_ID')
        self.kb_data_src_id = os.environ.get('KB_DATA_SRC_ID')
        if not self.kb_bucket or not self.kb_id or not self.kb_data_src_id:
            logger.error("KB_INPUT_BUCKET, KB_ID, or KB_DATA_SRC_ID env vars not set")
            raise ValueError("KB_INPUT_BUCKET, KB_ID and KB_DATA_SRC_ID environment variables are required.")
        logger.info(f"KB config: bucket={self.kb_bucket}, id={self.kb_id}, data_src_id={self.kb_data_src_id}")

    def list_questions(self, unanswered_only: bool = False) -> list[Question]:
        """
        Lists all questions from DynamoDB.
        :param unanswered_only: If true, only return unanswered questions.
        :return: A list of questions.
        """
        logger.info(f"Listing questions from DynamoDB, unanswered_only={unanswered_only}")
        query_args = {
            'KeyConditionExpression': boto3.dynamodb.conditions.Key('PK').eq('QUESTIONS')
        }
        if unanswered_only:
            query_args['FilterExpression'] = boto3.dynamodb.conditions.Attr('answer').not_exists()
        
        logger.info(f"Querying table with: {query_args}")
        try:
            response = self.table.query(**query_args)
            items = response.get('Items', [])
            logger.info(f"DynamoDB query returned {len(items)} items")
            return [Question(**item) for item in items]
        except Exception as e:
            logger.error(f"Error querying DynamoDB for questions: {e}", exc_info=True)
            raise

    def add_question(self, question: str, answer: str | None = None) -> Question:
        """
        Adds a new question to DynamoDB. The processed flag is set to False.

        :param question: The question text.
        :param answer: The optional answer to the question.
        :return: The created Question object.
        """
        new_question = Question(
            question=question,
            answer=answer
        )

        item_data = new_question.model_dump(exclude_none=True)
        item_data['PK'] = 'QUESTIONS'
        item_data['SK'] = new_question.question_id

        self.table.put_item(
            Item=item_data
        )
        return new_question

    def answer_question(self, question_id: str, answer: str) -> Question:
        """
        Adds an answer to a question in DynamoDB. The processed flag is set to False.

        :param question_id: The ID of the question to answer.
        :param answer: The answer to the question.
        :return: The updated question.
        :raises ValueError: if the question does not exist.
        """
        try:
            response = self.table.update_item(
                Key={'PK': 'QUESTIONS', 'SK': question_id},
                UpdateExpression="SET answer = :a, #processed = :p",
                ExpressionAttributeNames={
                    '#processed': 'processed',
                },
                ExpressionAttributeValues={
                    ':a': answer,
                    ':p': False,
                },
                ConditionExpression="attribute_exists(PK)",
                ReturnValues="ALL_NEW"
            )
            return Question(**response['Attributes'])
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                raise ValueError(f"Question with ID {question_id} not found.")
            else:
                raise

    def update_question(self, question_id: str, question: str | None = None, answer: str | None = None) -> Question:
        """
        Updates a question in DynamoDB. The question, answer or both can be updated.
        The processed flag is set to False after update.

        :param question_id: The ID of the question to update.
        :param question: The new question text.
        :param answer: The new answer to the question.
        :return: The updated Question object.
        :raises ValueError: if the question does not exist or no update data provided.
        """
        if question is None and answer is None:
            raise ValueError("Either question or answer must be provided for an update.")

        update_expression_parts = ['#processed = :p']
        expression_attribute_values = {':p': False}
        expression_attribute_names = {'#processed': 'processed'}

        if question is not None:
            update_expression_parts.append('question = :q')
            expression_attribute_values[':q'] = question

        if answer is not None:
            update_expression_parts.append('answer = :a')
            expression_attribute_values[':a'] = answer

        update_expression = "SET " + ", ".join(update_expression_parts)

        try:
            response = self.table.update_item(
                Key={'PK': 'QUESTIONS', 'SK': question_id},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attribute_values,
                ExpressionAttributeNames=expression_attribute_names,
                ConditionExpression="attribute_exists(PK)",
                ReturnValues="ALL_NEW"
            )
            return Question(**response['Attributes'])
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                raise ValueError(f"Question with ID {question_id} not found.")
            else:
                raise

    def delete_question(self, question_id: str):
        """
        Deletes a question from DynamoDB.

        :param question_id: The ID of the question to delete.
        :raises ValueError: if the question does not exist.
        """
        try:
            self.table.delete_item(
                Key={'PK': 'QUESTIONS', 'SK': question_id},
                ConditionExpression="attribute_exists(PK)"
            )
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                raise ValueError(f"Question with ID {question_id} not found.")
            else:
                raise

    def sync_to_knowledge_base(self) -> dict:
        """
        Syncs answered questions to the knowledge base.
        - Generates markdown from answered questions.
        - Uploads markdown to S3.
        - Triggers a Bedrock knowledge base sync.
        - Marks synced questions as processed.
        :return: A dictionary with the sync status and ingestion job details.
        """
        logger.info("Starting sync to knowledge base")
        try:
            markdown_content = self._create_markdown()

            if not markdown_content:
                logger.info("No answered questions with content to sync.")
                return {"status": "No answered questions to sync."}

            logger.info(f"Generated markdown content with length: {len(markdown_content)}")

            # Write to S3
            logger.info(f"Uploading markdown to S3 bucket {self.kb_bucket} with key qa/qa.md")
            self.s3_client.put_object(
                Bucket=self.kb_bucket,
                Key='qa/qa.md',
                Body=markdown_content.encode('utf-8')
            )
            logger.info("Successfully uploaded to S3")

            # Trigger Bedrock sync
            logger.info(f"Starting ingestion job for KB ID {self.kb_id} and data source ID {self.kb_data_src_id}")
            ingestion_job_response = self.bedrock_agent_client.start_ingestion_job(
                knowledgeBaseId=self.kb_id,
                dataSourceId=self.kb_data_src_id,
                clientToken=str(uuid.uuid4())
            )
            logger.info(f"Started ingestion job: {ingestion_job_response}")

            # Mark questions as processed
            logger.info("Marking questions as processed")
            answered_questions = [q for q in self.list_questions() if q.answer]
            with self.table.batch_writer() as batch:
                for q in answered_questions:
                    q.processed = True
                    item_data = q.model_dump(exclude_none=True)
                    item_data['PK'] = 'QUESTIONS'
                    item_data['SK'] = q.question_id
                    batch.put_item(Item=item_data)
            
            logger.info(f"Marked {len(answered_questions)} questions as processed.")

            return {
                "status": "Sync started",
                "ingestionJob": ingestion_job_response['ingestionJob']
            }
        except Exception as e:
            logger.error(f"Error during sync to knowledge base: {e}", exc_info=True)
            raise

    def _create_markdown(self) -> str:
        """
        Creates a markdown string from all answered questions.

        :return: A markdown string.
        """
        questions = self.list_questions()
        answered_questions = [q for q in questions if q.answer]

        markdown_parts = []
        for q in answered_questions:
            markdown_parts.append(f"# Question\n\n{q.question}\n\n# Answer\n\n{q.answer}")

        return "\n\n---\n\n".join(markdown_parts)
