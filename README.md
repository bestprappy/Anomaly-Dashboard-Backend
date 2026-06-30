# Team Todo

FastAPI + Postgres todo app with a built-in static frontend.

## Local Docker

Create `.env` with:

```env
POSTGRES_USER=todo
POSTGRES_PASSWORD=todo_secret
POSTGRES_DB=todos
API_PORT=8000
```

Then run:

```sh
docker compose up -d --build postgres api
```

Open `http://localhost:8000`.

## Render Deploy

This repo is ready for Render Blueprint deploys via `render.yaml`.

1. Push this repo to GitHub.
2. In Render, choose **New > Blueprint**.
3. Connect the GitHub repo.
4. Use the default Blueprint path: `render.yaml`.
5. Deploy the Blueprint.

Render will create:

- `todo-api`: Docker web service
- `todo-db`: Postgres database

The app URL will be the `todo-api` `onrender.com` URL.
