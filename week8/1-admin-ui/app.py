#!/usr/bin/env python3
import os

import aws_cdk as cdk

from kb.stack import KnowledgeBase
from twin.stack import Twin

env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region='us-west-2')
app = cdk.App()
kb = KnowledgeBase(app, "KnowledgeBaseStack", env=env)
_ = Twin(app, "Twin",
         kb_arn=kb.kb.knowledge_base_arn,
         kb_id=kb.kb.knowledge_base_id,
         kb_data_src_id=kb.kb.data_source_id,
         kb_input_bucket=kb.input_bucket,
         env=env,
         custom_domain_name="twin.numberwrangler.net",  # Replace with your domain
         custom_certificate_arn="arn:aws:acm:us-east-1:845422352941:certificate/cc28962c-4c16-46ee-ae43-f36c5c8b6a51"  # Replace with your certificate ARN
)

  

app.synth()
