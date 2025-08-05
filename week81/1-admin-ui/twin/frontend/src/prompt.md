
Create a single page index.html file which provides a professional and clean looking chat interface.
The index.html should use inline CSS and Javascript.
The chat is for a prospective employer to chat with a digital twin of the jobseeker.
Set the name of the jobseeker using a variable set at the top of the JavaScript, so that it will be easy
for someone to edit the name to use it for themself.

There are two endpoints to call, use relative paths:

- GET `/api/chat`

Returns a JSON object like this:

```json
{
  "messages": [
    {
      "role": "assistant",
      "content": [
        { "text": "Message from the assisstant" }
      ]
    },
    {
      "role": "user",
      "content": [
        { "text": "Message from the user" }
      ]
    }
  ]
}
```

- POST `/api/chat`

Send a message in an object like this:

```json
{ "prompt": "Who are you?"}
```

Returns a streaming response of `text/event-stream`
