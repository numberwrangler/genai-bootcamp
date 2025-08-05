# S3 Vector Custom CloudFormation Resource Lambda Function
import logging
import boto3
from botocore.exceptions import ClientError
import time

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize Boto3 client for S3 Vectors
s3vectors = boto3.client('s3vectors')

def create(event, context):
    """
    Handles the 'Create' event from CloudFormation.
    Creates an S3 Vector bucket and an index within it.
    """
    props = event['ResourceProperties']
    vector_bucket_name = props['VectorBucketName'].lower()
    vector_index_name = props['VectorIndexName']
    vector_dimension = int(props['VectorDimension'])

    logger.info(f"Creating vector bucket: {vector_bucket_name}")
    try:
        s3vectors.create_vector_bucket(vectorBucketName=vector_bucket_name)
        logger.info(f"✅ Vector bucket '{vector_bucket_name}' created successfully")
        
        response = s3vectors.get_vector_bucket(vectorBucketName=vector_bucket_name)
        bucket_info = response.get("vectorBucket", {})
        vector_store_arn = bucket_info.get("vectorBucketArn")
        
        if not vector_store_arn:
            raise ValueError("Vector bucket ARN not found in response")
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConflictException':
            logger.info(f"Bucket {vector_bucket_name} already exists. Proceeding to get ARN.")
            response = s3vectors.get_vector_bucket(vectorBucketName=vector_bucket_name)
            bucket_info = response.get("vectorBucket", {})
            vector_store_arn = bucket_info.get("vectorBucketArn")
        else:
            logger.error(f"❌ Error creating vector bucket: {e}")
            raise

    logger.info(f"Creating vector index: {vector_index_name} in bucket: {vector_bucket_name}")
    try:
        index_config = {
            "vectorBucketName": vector_bucket_name,
            "indexName": vector_index_name,
            "dimension": vector_dimension,
            "distanceMetric": "cosine",
            "dataType": "float32",
            "metadataConfiguration": {
                "nonFilterableMetadataKeys": ["AMAZON_BEDROCK_TEXT"]
            }
        }
        s3vectors.create_index(**index_config)
        logger.info(f"✅ Vector index '{vector_index_name}' creation initiated")

    except ClientError as e:
        if e.response['Error']['Code'] == 'ConflictException':
            logger.info(f"Index {vector_index_name} already exists. Proceeding...")
        else:
            logger.error(f"❌ Failed to create index: {e}")
            raise
    
    logger.info(f"Waiting for index '{vector_index_name}' to be available...")
    index_arn = None
    start_time = time.time()
    timeout = 60 # 1 minute timeout

    while time.time() - start_time < timeout:
        response = s3vectors.list_indexes(vectorBucketName=vector_bucket_name)
        for index in response.get("indexes", []):
            if index.get("indexName") == vector_index_name:
                index_arn = index.get("indexArn")
                break
        
        if index_arn:
            logger.info(f"Found index ARN: {index_arn}")
            break
        
        logger.info("Index not found yet, retrying in 5 seconds...")
        time.sleep(5)
        
    if not index_arn:
        raise ValueError(f"Index '{vector_index_name}' ARN not found after {timeout} seconds")

    return {
        'PhysicalResourceId': vector_bucket_name,
        'Data': {
            'VectorBucketArn': vector_store_arn,
            'IndexArn': index_arn
        }
    }

def update(event, context):
    """
    Handles the 'Update' event from CloudFormation.
    For this resource, updates are not supported. If properties change that
    require replacement, CloudFormation will issue a 'Create' for the new
    resource and a 'Delete' for the old one.
    """
    logger.info("Update event received. No action taken as updates are not supported for this resource.")
    return { 'PhysicalResourceId': event['PhysicalResourceId'] }

def delete(event, context):
    """
    Handles the 'Delete' event from CloudFormation.
    Deletes the S3 Vector index and then the bucket.
    """
    vector_bucket_name = event['PhysicalResourceId']
    props = event.get('ResourceProperties', {})
    vector_index_name = props.get('VectorIndexName')

    if not vector_index_name:
        logger.warning(f"VectorIndexName not found for bucket {vector_bucket_name}. Cannot delete index.")
    else:
        logger.info(f"Deleting index {vector_index_name} from bucket {vector_bucket_name}")
        try:
            s3vectors.delete_index(vectorBucketName=vector_bucket_name, indexName=vector_index_name)
            logger.info(f"✅ Index '{vector_index_name}' deleted successfully.")
        except s3vectors.exceptions.ResourceNotFoundException:
            logger.info(f"Index {vector_index_name} not found, skipping deletion.")
        except ClientError as e:
            logger.error(f"Error deleting index {vector_index_name}: {e}")

    logger.info(f"Deleting vector bucket {vector_bucket_name}")
    try:
        s3vectors.delete_vector_bucket(vectorBucketName=vector_bucket_name)
        logger.info(f"✅ Vector bucket '{vector_bucket_name}' deleted successfully.")
    except s3vectors.exceptions.ResourceNotFoundException:
        logger.info(f"Vector bucket {vector_bucket_name} not found, skipping deletion.")
    except ClientError as e:
        logger.error(f"Error deleting vector bucket {vector_bucket_name}: {e}")

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
