# Admin API Documentation

This document provides details on the available API endpoints for the Admin UI backend.

## Authentication

All endpoints require authentication. You must provide a valid Cognito JWT token in the `Authorization` header.

**Header:** `Authorization: Bearer <your_jwt_token>`

An invalid token will result in a `401 Unauthorized` response.

---

## Endpoints

### List Questions

- **Endpoint:** `GET /admin/api/questions`
- **Description:** Retrieves a list of questions.
- **Authentication:** Required.
- **Query Parameters:**
  - `unansweredOnly` (boolean, optional, default: `false`): If `true`, only questions without an answer are returned.
- **Success Response (`200 OK`):**
  - **Content:** `application/json`
  - **Body:** An array of question objects.
    ```json
    [
      {
        "question_id": "string",
        "question": "string",
        "answer": "string | null",
        "processed": "boolean"
      }
    ]
    ```

### Add Question

- **Endpoint:** `POST /admin/api/questions`
- **Description:** Adds a new question.
- **Authentication:** Required.
- **Request Body (`application/json`):**
  ```json
  {
    "question": "string",
    "answer": "string | null"
  }
  ```
- **Success Response (`201 Created`):**
  - **Content:** `application/json`
  - **Body:** The newly created question object.
    ```json
    {
      "question_id": "string",
      "question": "string",
      "answer": "string | null",
      "processed": false
    }
    ```

### Answer a Question

- **Endpoint:** `POST /admin/api/questions/{question_id}/answer`
- **Description:** Adds an answer to an existing question.
- **Authentication:** Required.
- **Path Parameters:**
  - `question_id` (string, required): The ID of the question to answer.
- **Request Body (`application/json`):**
  ```json
  {
    "answer": "string"
  }
  ```
- **Success Response (`200 OK`):**
  - **Content:** `application/json`
  - **Body:** The updated question object.
- **Error Response:**
  - `404 Not Found`: If the question ID does not exist.

### Update a Question

- **Endpoint:** `POST /admin/api/questions/{question_id}/update`
- **Description:** Updates a question's text and/or answer.
- **Authentication:** Required.
- **Path Parameters:**
  - `question_id` (string, required): The ID of the question to update.
- **Request Body (`application/json`):**
  ```json
  {
    "question": "string | null",
    "answer": "string | null"
  }
  ```
  At least one of `question` or `answer` must be provided.
- **Success Response (`200 OK`):**
  - **Content:** `application/json`
  - **Body:** The updated question object.
- **Error Responses:**
  - `400 Bad Request`: If neither `question` nor `answer` is provided.
  - `404 Not Found`: If the question ID does not exist.

### Delete a Question

- **Endpoint:** `DELETE /admin/api/questions/{question_id}`
- **Description:** Deletes a question.
- **Authentication:** Required.
- **Path Parameters:**
  - `question_id` (string, required): The ID of the question to delete.
- **Success Response (`204 No Content`):** The request was successful and there is no content to return.
- **Error Response:**
  - `404 Not Found`: If the question ID does not exist.

### Sync to Knowledge Base

- **Endpoint:** `POST /admin/api/sync`
- **Description:** Triggers an asynchronous sync of answered questions to the knowledge base.
- **Authentication:** Required.
- **Success Response (`202 Accepted`):**
  - **Content:** `application/json`
  - **Body:**
    ```json
    {
      "status": "string",
      "ingestionJob": {
        "knowledgeBaseId": "string",
        "dataSourceId": "string",
        "ingestionJobId": "string",
        "status": "string",
        "createdAt": "datetime",
        "updatedAt": "datetime"
      }
    }
    ```
    If there are no questions to sync, `ingestionJob` will be `null` and `status` will indicate this.
