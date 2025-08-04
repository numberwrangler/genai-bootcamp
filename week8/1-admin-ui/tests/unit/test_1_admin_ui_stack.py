import aws_cdk as core
import aws_cdk.assertions as assertions

from 1_admin_ui.1_admin_ui_stack import 1AdminUiStack

# example tests. To run these tests, uncomment this file along with the example
# resource in 1_admin_ui/1_admin_ui_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = 1AdminUiStack(app, "1-admin-ui")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
