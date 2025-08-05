import aws_cdk as cdk
import aws_cdk.aws_s3 as s3
from .vectorstore import S3VectorStore
from .bedrockkb import BedrockKnowledgeBase

class KnowledgeBase(cdk.Stack):
    kb: BedrockKnowledgeBase
    input_bucket: s3.Bucket
    def __init__(self, scope: cdk.App, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Bucket input documents will be uploaded to
        self.input_bucket = s3.Bucket(self, "InputBucket")

        vector_store = S3VectorStore(self, "VectorStore")

        self.kb = BedrockKnowledgeBase(self, "BedrockKb", 
            index_arn=vector_store.index_arn,
            vector_dimension=1024,
            input_bucket=self.input_bucket,
        )

        cdk.CfnOutput(self, "VectorBucketName", value=vector_store.vector_bucket_name)
        cdk.CfnOutput(self, "VectorIndexName", value=vector_store.vector_index_name)
        cdk.CfnOutput(self, "InputBucketName", value=self.input_bucket.bucket_name)
        cdk.CfnOutput(self, "KnowledgeBaseId", value=self.kb.knowledge_base_id)
        cdk.CfnOutput(self, "KnowledgeBaseArn", value=self.kb.knowledge_base_arn)
