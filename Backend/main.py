"""FastAPI app — the only HTTP surface for the poster agent.

Two endpoints:

    GET  /api/health    — liveness check
    POST /api/generate  — async, calls PosterAgent.arun(...)

Everything interesting lives in agent.py and agent_graph.py. This file is
just request validation + CORS + a thin async handler.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import poster_agent

app = FastAPI(title="Poster Agent API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    use_memory: bool = False


class GenerateResponse(BaseModel):
    session_id: str
    html: str
    css: str
    used_memory: bool
    history_length: int
    # Additive fields — the frontend can ignore them safely.
    rationale: Optional[str] = None
    palette: Optional[list[str]] = None


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "poster-agent"}


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    result = await poster_agent.arun(
        prompt=req.prompt,
        session_id=req.session_id,
        use_memory=req.use_memory,
    )
    return GenerateResponse(**result)


def main():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
