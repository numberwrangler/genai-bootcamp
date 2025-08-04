The Lambda function we are working on has all of its code under
`week8/1-admin-ui/twin/admin/src/app`. It is written in Python 3.13.
The code will be used under Lambda, with the `LambdaAdapterLayer` which
allows us to use FastAPI like a normal backend application.

The fastapi, pydantic and boto3 packages have already been added, as has PyJWT.

When creating code, create classes, group logically into separate files
and use import statements to reference.
Use built-in type available with Python 3.9 and above where possible.

A DynamoDB table is available - the name of the table is in the
environment variable `DDB_TABLE`. The partition key is named `PK`
and the range key is named `SK`.

# To do

- [x] Create a class for managing questions. We should store:
  - The question
  - An answer (optional)
  - Whether or not it has been processed (default to false)
- [x] Create a library class to validate incoming Cognito requests
  - Use PyJWT (already added to the project)
  - The user pool ID can by found at the environment variable `USER_POOL_ID`
  - The current region can be found from `AWS_REGION`
  - The JWKS can be found at `https://cognito-idp.<Region>.amazonaws.com/<userPoolId>/.well-known/jwks.json`
  - Load the JWKS at class init, so that we can re-use across invocations
  - Provide a function which takes the JWT as an argument and validates aainst the JWKS
- [x] Add a `GET` API endpoint which allows a user to list all questions
  - Ensure the request `Authorization` header is validated using the above class
  - Example header: `Authorization: Bearer abcdefg`
  - Accept an optional query string parameter 'unansweredOnly', which when true
    should filter results to only include unanswered questions
- [x] Add a POST API endpoint which allows a question to be answered
  - Ensure the request `Authorization` header is validated
- [x] Add a POST API endpoint to add a question
  - Ensure the request `Authorization` header is validated
  - Accept the question with or without an answer
  - The processed flag should be set to false
- [x] Add a POST API endpoint to update a question
  - The question, the answer or both can be updated
  - After update, the processed flag should be set to false
- [x] Add a DELETE API endpoint which allows a question to be deleted
  - Ensure the request `Authorization` header is validated
- [x] Add a private function to create markdown to the question management class
  - Return a single string with all questions/answers in markdown format
  - Do not include questions which have no answer
- [x] Add an API endpoint to sync questions to knowledge base
  - Generate markdown using above function
  - Write markdown to S3 bucket found from the env var `KB_INPUT_BUCKET` with the key `qa/qa.md`
  - Trigger a sync of the Bedrock Knowledge base. Use the env vars `KB_ID` and `KB_DATA_SRC_ID`
    for the knowledgeBaseId and dataSourceId
  - Mark all questions which have an answer recorded as processed
- [x] Document the API endpoints
  - Create a document at `week8/1-admin-ui/twin/admin/src/app/API.md` containing
    all necessary information for someone writing a client
  - Include all API endpoints outlined in this document
