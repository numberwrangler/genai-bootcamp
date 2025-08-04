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
        self.sns_client = boto3.client('sns')

        self.table_name = os.environ.get('DDB_TABLE')
        if not self.table_name:
            logger.error("DDB_TABLE environment variable is not set")
            raise ValueError("DDB_TABLE environment variable is required.")
        self.table = self.dynamodb.Table(self.table_name)
        logger.info(f"DynamoDB table name: {self.table_name}")

        self.notification_topic_arn = os.environ.get('NOTIFICATION_TOPIC_ARN')
        if self.notification_topic_arn:
            logger.info(f"SNS notification topic ARN: {self.notification_topic_arn}")
        else:
            logger.warning("NOTIFICATION_TOPIC_ARN environment variable is not set. SNS notifications will be disabled.")

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

        if self.notification_topic_arn:
            try:
                message = f"A new question has been added:\n\n{question}"
                self.sns_client.publish(
                    TopicArn=self.notification_topic_arn,
                    Message=message,
                    Subject="New Question Added"
                )
                logger.info(f"Published new question notification to SNS topic {self.notification_topic_arn}")
            except Exception as e:
                # Do not fail the request if SNS notification fails
                logger.error(f"Failed to publish to SNS topic {self.notification_topic_arn}: {e}", exc_info=True)

        return new_question

