# AI Poster Studio

A small full-stack playground where a user describes a poster, an AI agent
generates HTML/CSS for it, and the user can iterate via follow-ups (with
session memory) or start over.

The current backend ships **mock** poster output so the frontend contract can
be wired end-to-end before the real agent is plugged in.

## Folder structure

```
Learning/
├── Backend/                  # FastAPI service (Python)
│   ├── main.py               # API: /api/health, /api/generate
│   ├── llm_client.py         # litellm helper (Gemini / Ollama)
│   ├── db_client.py          # Postgres helper (psycopg)
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── .env                  # GEMINI_API_KEY, DB_*, LOCAL_MODEL
├── Frontend/
│   └── my-app/               # Vite + React 19 app
│       ├── index.html
│       ├── package.json
│       └── src/
│           ├── App.jsx                # top-level view switcher
│           ├── api/posterApi.js       # fetch wrapper for backend
│           ├── hooks/usePosterAgent.js# session/state hook
│           ├── components/
│           │   ├── PosterForm.jsx     # initial brief form
│           │   ├── PosterPreview.jsx  # iframe renderer for HTML/CSS
│           │   └── FollowUpPanel.jsx  # follow-up + generate-new controls
│           ├── styles/app.css
│           ├── index.css
│           └── main.jsx
├── docker-compose.yml        # Postgres + pgAdmin for local dev
└── README.md
```

## How it works

1. **Initial view** — `PosterForm` collects title, audience, style, and free-form
   details and POSTs them to `/api/generate` with `use_memory: false`.
2. The backend returns `{ session_id, html, css, used_memory, history_length }`.
3. **Result view** — split layout:
   - **Left half** — `PosterPreview` renders the returned HTML/CSS inside a
     sandboxed `<iframe>` (via `srcdoc`) so the poster styles can't leak into
     the app shell.
   - **Right half** — `FollowUpPanel` shows the brief history plus a textarea.
4. **Follow up** — sends the new prompt with the current `session_id` and
   `use_memory: true`. The backend appends to the session's history (the real
   agent will use that as context).
5. **Generate New** — clears state and returns to the form view, starting a
   fresh session on the next submit.

## Backend

### Setup

```bash
cd Backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
# from Backend/
uvicorn main:app --reload --port 8000
# or
python main.py
```

### Endpoints

| Method | Path            | Purpose                                          |
| ------ | --------------- | ------------------------------------------------ |
| GET    | `/api/health`   | Test endpoint — returns `{"status":"ok",...}`    |
| POST   | `/api/generate` | Generate a poster (mock HTML/CSS for now)        |

**Request** to `/api/generate`:

```json
{
  "prompt": "Title: Summer Festival | Audience: Students | Style: Bold retro",
  "session_id": null,
  "use_memory": false
}
```

**Response**:

```json
{
  "session_id": "8b3f...",
  "html": "<div class=\"poster\">…</div>",
  "css": "* { box-sizing: border-box; } …",
  "used_memory": false,
  "history_length": 1
}
```

For follow-ups, send the same `session_id` back with `use_memory: true`. The
backend keeps an in-memory `SESSIONS` dict so the real agent can later read
prior turns as context.

## Frontend

### Setup

```bash
cd Frontend/my-app
npm install
```

### Run

```bash
npm run dev      # http://localhost:5173
npm run build    # production build into dist/
npm run lint
```

The API base URL defaults to `http://localhost:8000`. Override with
`VITE_API_BASE` in `Frontend/my-app/.env` if needed.

## Database (optional)

`docker-compose.yml` brings up Postgres + pgAdmin for when the backend grows
beyond mock responses:

```bash
docker compose up -d
# pgAdmin: http://localhost:5050
# Postgres: localhost:5432
```

## Roadmap

- Replace `_mock_poster()` in `Backend/main.py` with a call to `llm_client.py`
  that prompts an LLM to emit `{html, css}`.
- Persist sessions in Postgres (via `db_client.py`) instead of the in-memory
  dict.
- Add a "download as image" action in `PosterPreview`.
