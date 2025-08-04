We are working on a SvelteKit 5 SPA written in TypeScript.
All files should be under `week8/1-admin-ui/twin/frontend/admin`.

When creating code, create separate libraries and group logically into
files.

All API endpoints should be referenced unqualified as it runs on the same host.

We are using Cognito for authentication with the `@aws-amplify/auth` package
already installed from npm.

The following environment variables are available at build time and can be accessed
via `import.meta.env`:

- `PUBLIC_VITE_COGNITO_USER_POOL_ID`
- `PUBLIC_VITE_COGNITO_USER_POOL_CLIENT_ID`

When building the UI, spend the time to make it look clean, crisp and professional.
The lucide-svelte icon toolkit is available and should be used.
Tailwind CSS is available and should be used
Reference the API documentation at `week8/1-admin-ui/twin/admin/src/app/API.md`

# To Do

## Library functions

Put these is a sensible location where they can be referenced from view code.
All requests to the admin API

 - [x] List questions
 - [x] Add questions
 - [x] Answer questions
 - [x] Update questions
 - [x] Delete questions
 - [x] Sync to knowledge base

## Views

- [x] Add a login page

We are using Cognito passwordless logins:

    User enters their email address to sign up/sign in
    They receive an email message with a time-limited code
    After the users enters their code they are authenticated

- [x] Add a question management page that will:
    - List questions (with an option to filter for unanswered)
    - Allow adding, editing and deleting questions
    - A button to sync to the knowledge base

