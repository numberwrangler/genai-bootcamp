import aws_cdk as cdk
from constructs import Construct
import aws_cdk.aws_lambda as _lambda
import aws_cdk.aws_iam as iam
import aws_cdk.custom_resources as cr
from aws_cdk import BundlingOptions


class S3VectorStore(Construct):
    bucket_arn: str
    index_arn: str
    vector_bucket_name: str
    vector_index_name: str

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Custom resource to create S3 Vector bucket and index
        s3vector_fn = _lambda.Function(
            self,
            "S3VectorCustomResourceHandler",
            architecture=_lambda.Architecture.X86_64,
            runtime=_lambda.Runtime.PYTHON_3_13,
            handler="main.handler",
            code=_lambda.Code.from_asset("kb/s3vector",
                bundling=BundlingOptions(
                    image=_lambda.Runtime.PYTHON_3_13.bundling_image,
                    command=[
                        'bash', '-c',
                        'pip install uv && uv export --frozen --no-dev --no-editable -o requirements.txt && pip install -r requirements.txt -t /asset-output && cp -r *py /asset-output/'
                    ],
                    user='root'
                )
            ),
            timeout=cdk.Duration.minutes(1),
        )

        s3vector_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3vectors:*"],
                resources=["*"],
            )
        )

        provider = cr.Provider(
            self,
            "S3VectorProvider",
            on_event_handler=s3vector_fn,
        )

        self.vector_bucket_name = cdk.Fn.join("-", [
            cdk.Aws.STACK_NAME,
            cdk.Aws.ACCOUNT_ID,
            cdk.Aws.REGION,
        ])
        self.vector_index_name = "document-index"
        vector_dimension = 1024

        s3_vector_store_cr = cdk.CustomResource(
            self,
            "S3VectorStoreCustomResource",
            service_token=provider.service_token,
            properties={
                "VectorBucketName": self.vector_bucket_name,
                "VectorIndexName": self.vector_index_name,
                "VectorDimension": vector_dimension,
            }
        )

        self.bucket_arn = s3_vector_store_cr.get_att_string("VectorBucketArn")
        self.index_arn = s3_vector_store_cr.get_att_string("IndexArn")
