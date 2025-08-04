"""
RUM - Real User Monitoring
"""
from constructs import Construct
from aws_cdk import Stack
import aws_cdk.aws_rum as rum
import json


class Rum(Construct):
    def __init__(self, scope: Construct, id: str, domain_name: str) -> None:
        super().__init__(scope, id)

        app_name = "twin"

        resource_policy_json = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "rum:PutRumEvents",
                    "Resource": f"arn:aws:rum:{Stack.of(self).region}:{Stack.of(self).account}:appmonitor/{app_name}"
                }
            ]
        }

        app_monitor = rum.CfnAppMonitor(
            self, "AppMonitor",
            name=app_name,
            domain=domain_name,
            app_monitor_configuration=rum.CfnAppMonitor.AppMonitorConfigurationProperty(
                allow_cookies=True,
                session_sample_rate=1,
                telemetries=['errors', 'performance', 'http'],

            ),
            cw_log_enabled=True,
            resource_policy=rum.CfnAppMonitor.ResourcePolicyProperty(
                policy_document=json.dumps(resource_policy_json),
            ),
        )

