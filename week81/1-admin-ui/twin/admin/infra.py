from aws_cdk import RemovalPolicy
from aws_cdk.aws_s3 import Bucket
from constructs import Construct
import aws_cdk.aws_dynamodb as dynamodb
import aws_cdk.aws_lambda as _lambda

from .cognito import Cognito
from .apifn import ApiFn


class Admin(Construct):
    dynamodb_table: dynamodb.TableV2
    endpoint: str
    domain_name: str
    user_pool_id: str
    user_pool_client_id: str


    def __init__(self, scope: Construct, id: str,
                   kb_id: str,  # Bedrock Knowledge Base ID
                   kb_data_src_id: str,  # Bedrock Knowledge Base Data Source ID
                   kb_input_bucket: Bucket, # S3 Bucket for input documents
                 ) -> None:
        super().__init__(scope, id)

        # Database used to store settings, pending questions etc
        self.dynamodb_table = dynamodb.TableV2(self, 'AdminTable',
                                          partition_key=dynamodb.Attribute(name='PK', type=dynamodb.AttributeType.STRING),
                                          sort_key=dynamodb.Attribute(name='SK', type=dynamodb.AttributeType.STRING),
                                          removal_policy=RemovalPolicy.DESTROY,
                                        )

        cognito = Cognito(self, 'Cognito')
        self.user_pool_id = cognito.user_pool.user_pool_id
        self.user_pool_client_id = cognito.client.user_pool_client_id

        api_fn = ApiFn(self, 'ApiFn',
                       dynamo_db_table=self.dynamodb_table,
                       kb_id=kb_id,
                       kb_data_src_id=kb_data_src_id,
                       kb_input_bucket=kb_input_bucket,
                       user_pool_id=cognito.user_pool.user_pool_id,
                      )

        self.endpoint = api_fn.endpoint
        self.domain_name = api_fn.domain_name
