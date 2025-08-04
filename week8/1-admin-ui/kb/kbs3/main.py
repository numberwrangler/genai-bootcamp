# Knowledge Base Custom CloudFormation Resource Lambda Function
import logging
import boto3
from botocore.exceptions import ClientError
import time
import os

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize Boto3 client for Bedrock Agent
bedrock = boto3.client('bedrock-agent')
region_name = os.environ['AWS_REGION']

def create(event, context):
    """
    Handles the 'Create' event from CloudFormation.
    Creates a Bedrock Knowledge Base.
    """
    props = event['ResourceProperties']
    role_arn = props['RoleArn']
    index_arn = props['IndexArn']
    vector_dimension = int(props['VectorDimension'])
    stack_name = props['StackName']

    # Define Knowledge Base name with unique identifier
    kb_name = f"kb-s3-vectors-{stack_name.lower()}"

    logger.info(f"Creating knowledge base: {kb_name}")

    try:
        # Create the Knowledge Base
        create_kb_response = bedrock.create_knowledge_base(
            name=kb_name,
            description='Amazon Bedrock Knowledge Bases with S3 Vector Store',
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                'type': 'VECTOR',
                'vectorKnowledgeBaseConfiguration': {
                    'embeddingModelArn': f'arn:aws:bedrock:{region_name}::foundation-model/amazon.titan-embed-text-v2:0',
                    'embeddingModelConfiguration': {
                        'bedrockEmbeddingModelConfiguration': {
                            'dimensions': vector_dimension,
                            'embeddingDataType': 'FLOAT32'
                        }
                    },
                },
            },
            storageConfiguration={
                'type': 'S3_VECTORS',
                's3VectorsConfiguration': {
                    'indexArn': index_arn,
                },
            }
        )

        knowledge_base_id = create_kb_response["knowledgeBase"]["knowledgeBaseId"]
        knowledge_base_arn = create_kb_response["knowledgeBase"]["knowledgeBaseArn"]
        logger.info(f"Knowledge base ID: {knowledge_base_id}")

        logger.info(f"Waiting for knowledge base {knowledge_base_id} to finish creating...")

        # Poll for KB creation status
        status = "CREATING"
        start_time = time.time()

        while status == "CREATING":
            response = bedrock.get_knowledge_base(knowledgeBaseId=knowledge_base_id)
            status = response['knowledgeBase']['status']
            elapsed_time = int(time.time() - start_time)
            logger.info(f"Current status: {status} (elapsed time: {elapsed_time}s)")
            
            if status == "CREATING":
                logger.info("Still creating, checking again in 30 seconds...")
                time.sleep(30)
            else:
                break
        
        if status != "ACTIVE":
            raise Exception(f"Knowledge base creation failed with status: {status}")

        logger.info(f"✅ Knowledge base creation completed with status: {status}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConflictException':
            logger.info(f"Knowledge base {kb_name} already exists. Trying to find it.")
            response = bedrock.list_knowledge_bases(maxResults=100)
            for kb in response.get("knowledgeBaseSummaries", []):
                if kb.get("name") == kb_name:
                    knowledge_base_id = kb.get("knowledgeBaseId")
                    kb_details = bedrock.get_knowledge_base(knowledgeBaseId=knowledge_base_id)
                    knowledge_base_arn = kb_details['knowledgeBase']['knowledgeBaseArn']
                    logger.info(f"Found existing knowledge base with ID: {knowledge_base_id}")
                    break
            if not knowledge_base_id:
                 raise Exception(f"Could not find existing knowledge base {kb_name}")
        else:
            logger.error(f"❌ Error creating knowledge base: {e}")
            raise

    return {
        'PhysicalResourceId': knowledge_base_id,
        'Data': {
            'KnowledgeBaseId': knowledge_base_id,
            'KnowledgeBaseArn': knowledge_base_arn
        }
    }

def update(event, context):
    """
    Handles the 'Update' event from CloudFormation.
    For this resource, updates are not supported.
    """
    logger.info("Update event received. No action taken as updates are not supported for this resource.")
    return { 'PhysicalResourceId': event['PhysicalResourceId'] }

def delete(event, context):
    """
    Handles the 'Delete' event from CloudFormation.
    Deletes the Bedrock Knowledge Base.
    """
    knowledge_base_id = event['PhysicalResourceId']
    
    logger.info(f"Deleting knowledge base {knowledge_base_id}")
    try:
        bedrock.delete_knowledge_base(knowledgeBaseId=knowledge_base_id)
        
        logger.info(f"Waiting for knowledge base {knowledge_base_id} to be deleted...")

        status = "DELETING"
        start_time = time.time()
        
        while status == "DELETING":
            try:
                response = bedrock.get_knowledge_base(knowledgeBaseId=knowledge_base_id)
                status = response['knowledgeBase']['status']
                elapsed_time = int(time.time() - start_time)
                logger.info(f"Current status: {status} (elapsed time: {elapsed_time}s)")
                if status == "DELETING":
                    logger.info("Still deleting, checking again in 30 seconds...")
                    time.sleep(30)
                else:
                    break
            except bedrock.exceptions.ResourceNotFoundException:
                logger.info("Knowledge base not found, assuming it is deleted.")
                status = "DELETED"
                break

        logger.info(f"✅ Knowledge base deletion completed.")
        
    except bedrock.exceptions.ResourceNotFoundException:
        logger.info(f"Knowledge base {knowledge_base_id} not found, skipping deletion.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConflictException':
             logger.info(f"Knowledge base {knowledge_base_id} already being deleted.")
        else:
            logger.error(f"Error deleting knowledge base {knowledge_base_id}: {e}")

    return { 'PhysicalResourceId': event['PhysicalResourceId'] }

def handler(event, context):
    """
    Lambda handler function that dispatches events to the appropriate function.
    """
    request_type = event['RequestType']
    logger.info(f"Received request: {request_type}")

    if request_type == 'Create':
        return create(event, context)
    if request_type == 'Update':
        return update(event, context)
    if request_type == 'Delete':
        return delete(event, context)
    
    raise Exception(f"Invalid request type: {request_type}")
