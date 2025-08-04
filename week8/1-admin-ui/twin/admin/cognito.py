from constructs import Construct
import aws_cdk.aws_cognito as cognito


class Cognito(Construct):
    user_pool: cognito.UserPool
    client: cognito.UserPoolClient

    def __init__(self, scope: Construct, id: str) -> None:
        super().__init__(scope, id)

        self.user_pool = cognito.UserPool(
            self,
            "UserPool",
            user_pool_name="admin-ui-user-pool",
            self_sign_up_enabled=True,
            sign_in_aliases=cognito.SignInAliases(email=True),
            custom_attributes={
                "isAdmin": cognito.BooleanAttribute(),
            },
            sign_in_policy=cognito.SignInPolicy(
                allowed_first_auth_factors=cognito.AllowedFirstAuthFactors(password=True, email_otp=True),
            ),
        )

        self.client = self.user_pool.add_client(
            "UserPoolClient",
            auth_flows=cognito.AuthFlow(
                user=True,
                user_srp=True,
            ),
            supported_identity_providers=[cognito.UserPoolClientIdentityProvider.COGNITO],
        )

