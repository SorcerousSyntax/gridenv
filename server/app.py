"""
GridWorld Survival — FastAPI server for HuggingFace Spaces.
"""
import uuid
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from gridenv import GridWorldEnv

_sessions: Dict[str, GridWorldEnv] = {}

app = FastAPI(title="GridWorld Survival", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class CreateReq(BaseModel):
    task_id: str = "reach_goal"
    seed: Optional[int] = None

class StepReq(BaseModel):
    session_id: str
    action: str


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(LANDING)

@app.post("/reset")
async def reset(req: Optional[CreateReq] = None):
    try:
        task_id = req.task_id if req else "reach_goal"
        seed = req.seed if req else None
        env = GridWorldEnv(task_id=task_id, seed=seed)
        obs = env.reset()
        sid = str(uuid.uuid4())
        _sessions[sid] = env
        return {"session_id": sid, "observation": obs.dict()}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/step")
async def step(req: StepReq):
    env = _sessions.get(req.session_id)
    if not env:
        raise HTTPException(404, "Session not found. Call /reset first.")
    obs, reward, done, info = env.step(req.action)
    return {"observation": obs.dict(), "reward": reward.dict(), "done": done, "info": info}

@app.get("/state")
async def get_state(session_id: str):
    env = _sessions.get(session_id)
    if not env:
        raise HTTPException(404, "Session not found.")
    return env.state().dict()

@app.post("/sessions/create")
async def create_session(req: CreateReq):
    return await reset(req)

@app.post("/sessions/step")
async def session_step(req: StepReq):
    return await step(req)

@app.get("/sessions/state")
async def session_state(session_id: str):
    return await get_state(session_id)

@app.get("/tasks")
async def tasks():
    return {"tasks": [
        {"id": "reach_goal",         "difficulty": "easy",   "max_steps": 30},
        {"id": "collect_and_escape", "difficulty": "medium", "max_steps": 50},
        {"id": "survive_and_escape", "difficulty": "hard",   "max_steps": 80},
    ]}

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "sessions": len(_sessions)}


LANDING = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GridWorld Survival</title>
</head>
<body>
<h1>GridWorld Survival</h1>
</body>
</html>"""


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()