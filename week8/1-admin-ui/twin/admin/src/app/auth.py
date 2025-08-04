import os
import jwt
from jwt import PyJWKClient
import logging

logger = logging.getLogger(__name__)


class CognitoValidator:
    """
    Validates a Cognito JWT token.
    """
    def __init__(self):
        """
        Initialises the validator.
        - Fetches the user pool ID and region from environment variables.
        - Constructs the JWKS URL.
        - Creates a PyJWKClient to handle fetching and caching of JWKS.
        """
        self.user_pool_id = os.environ.get("USER_POOL_ID")
        self.region = os.environ.get("AWS_REGION")
        if not self.user_pool_id or not self.region:
            logger.error("USER_POOL_ID and/or AWS_REGION environment variables are not set")
            raise ValueError("USER_POOL_ID and AWS_REGION environment variables are required.")

        logger.info(f"CognitoValidator initialized with region: {self.region}, user pool id: {self.user_pool_id}")
        self.jwks_url = f"https://cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}/.well-known/jwks.json"
        self.issuer = f"https://cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}"
        self.jwks_client = PyJWKClient(self.jwks_url)

    def validate_token(self, token: str) -> dict:
        """
        Validates the JWT token against the JWKS.
        :param token: The JWT token to validate.
        :return: The decoded token payload.
        :raises ValueError: If the token is invalid.
        """
        try:
            logger.info("Getting signing key from JWT")
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            logger.info("Signing key obtained, decoding token")

            data = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self.issuer,
                options={"verify_exp": True, "verify_at_hash": False, "verify_aud": False},
            )
            logger.info("Token decoded successfully")
            return data
        except jwt.exceptions.InvalidTokenError as e:
            logger.error(f"JWT validation failed: {e}", exc_info=True)
            raise ValueError(f"Invalid token: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during token validation: {e}", exc_info=True)
            raise ValueError(f"An unexpected error occurred during token validation.") from e
