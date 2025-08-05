from aws_cdk.aws_dynamodb import TableV2
from aws_cdk.aws_s3 import Bucket
from aws_cdk import aws_iam as iam
from aws_cdk import Duration, BundlingOptions, Fn, Stack
from constructs import Construct
import aws_cdk.aws_lambda as _lambda


class ApiFn(Construct):
    """
    Lambda function to handle API requests for the admin UI.
    """
    endpoint: str
    domain_name: str

    def __init__(self, scope: Construct, id: str,
                 dynamo_db_table: TableV2,
                 kb_id: str,
                 kb_data_src_id: str,
                 kb_input_bucket: Bucket,
                 user_pool_id: str,
                )-> None:
        super().__init__(scope, id)

        fn = _lambda.Function(
            self,
            "AdminApiFunction",
            runtime=_lambda.Runtime.PYTHON_3_13,
            architecture=_lambda.Architecture.X86_64,
            handler="run.sh",
            code=_lambda.Code.from_asset("twin/admin/src",
                                         bundling=BundlingOptions(
                                             image=_lambda.Runtime.PYTHON_3_13.bundling_image,
                                             command=[
                                                 "bash", "-c",
                                                 "pip install uv && uv export --frozen --no-dev --no-editable -o requirements.txt && pip install -r requirements.txt -t /asset-output && cp -r app/* /asset-output/"
                                             ],
                                             user="root",
                                             platform="linux/amd64",
                                         )),
            timeout=Duration.seconds(10),
            layers=[
                _lambda.LayerVersion.from_layer_version_arn(
                    self,
                    "LambdaAdapterLayer",
                    f"arn:aws:lambda:{Stack.of(self).region}:753240598075:layer:LambdaAdapterLayerX86:25"
                )
            ],
            environment={
                "AWS_LAMBDA_EXEC_WRAPPER": "/opt/bootstrap",
                "DDB_TABLE": dynamo_db_table.table_name,
                "KB_DATA_SRC_ID": kb_data_src_id,
                "KB_ID": kb_id,
                "KB_INPUT_BUCKET": kb_input_bucket.bucket_name,
                "PORT": "8000",
                "USER_POOL_ID": user_pool_id,
            },
            memory_size=256,
        )
        _ = kb_input_bucket.grant_read_write(fn)
        _ = dynamo_db_table.grant_read_write_data(fn)
        fn.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:ListKnowledgeBases",
                    "bedrock:ListDataSources",
                    "bedrock:UpdateKnowledgeBase",
                    "bedrock:UpdateDataSource",
                    "bedrock:StartIngestionJob",
                ],
                resources=["*"],  # Adjust as necessary for security
            )
        )


        fn_url = fn.add_function_url(
                auth_type=_lambda.FunctionUrlAuthType.NONE,  # Cognito auth in handled in the code base
                invoke_mode=_lambda.InvokeMode.BUFFERED,
                cors=_lambda.FunctionUrlCorsOptions(
                    allowed_methods=[_lambda.HttpMethod.ALL],
                    allowed_origins=["*"],
                ),
            )
        self.endpoint = fn_url.url
        self.domain_name = Fn.select(2, Fn.split("/", self.endpoint)) # Remove the protocol part from the URL
