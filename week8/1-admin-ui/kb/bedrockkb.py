import aws_cdk as cdk
from constructs import Construct
import aws_cdk.aws_lambda as _lambda
import aws_cdk.aws_iam as iam
import aws_cdk.custom_resources as cr
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_bedrock as bedrock
from aws_cdk import BundlingOptions

class BedrockKnowledgeBase(Construct):
    knowledge_base_id: str
    knowledge_base_arn: str
    data_source_id: str

    def __init__(self, scope: Construct, id: str, 
                 index_arn: str,
                 vector_dimension: int,
                 input_bucket: s3.Bucket,
                 **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Role for the Knowledge Base
        kb_role = iam.Role(
            self,
            "KnowledgeBaseRole",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
        )
        kb_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[f"arn:aws:bedrock:{cdk.Aws.REGION}::foundation-model/*"],
            )
        )
        kb_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3vectors:*"],
                resources=["*"],
            )
        )

        input_bucket.grant_read(kb_role)

        # Custom resource to create Bedrock Knowledge Base
        kbs3_fn = _lambda.Function(
            self,
            "Kbs3CustomResourceHandler",
            architecture=_lambda.Architecture.X86_64,
            runtime=_lambda.Runtime.PYTHON_3_13,
            handler="main.handler",
            code=_lambda.Code.from_asset("kb/kbs3",
                bundling=BundlingOptions(
                    image=_lambda.Runtime.PYTHON_3_13.bundling_image,
                    command=[
                        'bash', '-c',
                        'pip install uv && uv export --frozen --no-dev --no-editable -o requirements.txt && pip install -r requirements.txt -t /asset-output && cp -r *py /asset-output/'
                    ],
                    user='root'
                )
            ),
            timeout=cdk.Duration.minutes(10), # KB creation can take time
        )
        
        kbs3_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:CreateKnowledgeBase",
                    "bedrock:GetKnowledgeBase",
                    "bedrock:ListKnowledgeBases",
                    "bedrock:DeleteKnowledgeBase"
                ],
                resources=[f"arn:aws:bedrock:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:knowledge-base/*"],
            )
        )
        kbs3_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[kb_role.role_arn],
            )
        )

        provider = cr.Provider(
            self,
            "Kbs3Provider",
            on_event_handler=kbs3_fn,
        )

        kb_cr = cdk.CustomResource(
            self,
            "Kbs3CustomResource",
            service_token=provider.service_token,
            properties={
                "RoleArn": kb_role.role_arn,
                "IndexArn": index_arn,
                "VectorDimension": vector_dimension,
                "StackName": cdk.Aws.STACK_NAME
            }
        )

        self.knowledge_base_id = kb_cr.get_att_string("KnowledgeBaseId")
        self.knowledge_base_arn = kb_cr.get_att_string("KnowledgeBaseArn")

        data_source = bedrock.CfnDataSource(self, "DataSource",
            knowledge_base_id=self.knowledge_base_id,
            name=input_bucket.bucket_name,
            data_source_configuration=bedrock.CfnDataSource.DataSourceConfigurationProperty(
                s3_configuration=bedrock.CfnDataSource.S3DataSourceConfigurationProperty(
                    bucket_arn=input_bucket.bucket_arn
                ),
                type="S3"
            ),
            vector_ingestion_configuration=bedrock.CfnDataSource.VectorIngestionConfigurationProperty(
                chunking_configuration=bedrock.CfnDataSource.ChunkingConfigurationProperty(
                    chunking_strategy="SEMANTIC",
                    semantic_chunking_configuration=bedrock.CfnDataSource.SemanticChunkingConfigurationProperty(
                        breakpoint_percentile_threshold=95,
                        buffer_size=1,
                        max_tokens=512,
                    ),
                )
            )
        )
        self.data_source_id = data_source.get_att("DataSourceId").to_string()
