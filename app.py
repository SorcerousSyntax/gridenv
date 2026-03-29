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
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
  :root { --bg:#0a0a0f; --panel:#12121a; --accent:#00ff88; --warn:#ff6b35; --text:#e0e0e0; --dim:#555; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Rajdhani',sans-serif; min-height:100vh; }
  header { border-bottom:1px solid var(--accent); padding:1.5rem 3rem; display:flex; justify-content:space-between; align-items:center; }
  header h1 { font-size:1.8rem; font-weight:700; color:var(--accent); letter-spacing:0.1em; }
  header span { font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:var(--dim); }
  main { max-width:860px; margin:3rem auto; padding:0 2rem; }
  .tasks { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:2rem 0; }
  .task { border:1px solid var(--dim); padding:1.2rem; background:var(--panel); }
  .task .badge { font-family:'Share Tech Mono',monospace; font-size:0.7rem; padding:0.2rem 0.5rem; display:inline-block; margin-bottom:0.6rem; }
  .easy .badge { background:#1a3a2a; color:#00ff88; border:1px solid #00ff88; }
  .medium .badge { background:#3a2a0a; color:#ffaa00; border:1px solid #ffaa00; }
  .hard .badge { background:#3a0a0a; color:#ff4444; border:1px solid #ff4444; }
  .task h3 { font-size:1rem; font-weight:600; margin-bottom:0.4rem; }
  .task p { font-size:0.82rem; color:#888; line-height:1.5; }
  .grid-demo { font-family:'Share Tech Mono',monospace; font-size:1rem; line-height:1.6; background:var(--panel); border:1px solid var(--dim); padding:1rem 1.5rem; margin:1.5rem 0; color:var(--accent); }
  .ep { color:#00ff88; } .wall { color:#333; } .coin { color:#ffaa00; } .trap { color:#ff4444; } .enemy { color:#ff6b35; } .goal { color:#00aaff; }
  .endpoints { margin:1.5rem 0; }
  .ep-row { background:var(--panel); border:1px solid var(--dim); padding:0.7rem 1rem; margin:0.4rem 0; font-family:'Share Tech Mono',monospace; font-size:0.82rem; display:flex; gap:1rem; }
  .method { color:var(--accent); font-weight:600; }
  .links { display:flex; gap:1rem; margin:1.5rem 0; }
  .links a { border:1px solid var(--accent); padding:0.5rem 1.2rem; text-decoration:none; color:var(--accent); font-size:0.9rem; font-weight:600; letter-spacing:0.05em; }
  .links a:hover { background:var(--accent); color:var(--bg); }
  h2 { font-size:1.2rem; font-weight:700; letter-spacing:0.05em; margin:2rem 0 0.8rem; color:var(--accent); text-transform:uppercase; }
  footer { border-top:1px solid #1a1a2a; padding:1.5rem 3rem; text-align:center; font-size:0.78rem; color:var(--dim); margin-top:3rem; }
</style>
</head>
<body>
<header>
  <h1>⬛ GRIDWORLD SURVIVAL</h1>
  <span>OpenEnv v1.0.0 · Mini-Game RL Environment</span>
</header>
<main>
  <p style="line-height:1.7;color:#aaa;margin-bottom:1rem;">
    A procedurally generated 2D grid mini-game for AI agent training. 
    Navigate hazards, collect coins, avoid enemies, and escape.
    Three tasks with increasing difficulty.
  </p>

  <div class="grid-demo">
+---------------+<br>
|<span class="ep">A</span> . . <span class="wall">#</span> . . .|<br>
|. <span class="wall">#</span> . . <span class="coin">C</span> . .|<br>
|. . <span class="trap">T</span> <span class="wall">#</span> . . .|<br>
|<span class="coin">C</span> . . . <span class="enemy">E</span> . .|<br>
|. <span class="wall">#</span> . . . . <span class="goal">G</span>|<br>
+---------------+
  </div>

  <h2>Tasks</h2>
  <div class="tasks">
    <div class="task easy"><span class="badge">EASY</span><h3>Reach the Goal</h3><p>5×5 grid. Navigate from A to G. No hazards.</p></div>
    <div class="task medium"><span class="badge">MEDIUM</span><h3>Collect & Escape</h3><p>7×7 grid. Collect 3+ coins then reach G.</p></div>
    <div class="task hard"><span class="badge">HARD</span><h3>Survive & Escape</h3><p>9×9 grid. Enemies + traps + coins + escape.</p></div>
  </div>

  <h2>API</h2>
  <div class="endpoints">
    <div class="ep-row"><span class="method">POST</span> /reset — start new episode</div>
    <div class="ep-row"><span class="method">POST</span> /step  — send action (UP/DOWN/LEFT/RIGHT/STAY)</div>
    <div class="ep-row"><span class="method">GET</span>  /state — get full game state</div>
    <div class="ep-row"><span class="method">GET</span>  /tasks — list all tasks</div>
  </div>

  <div class="links">
    <a href="/docs">API DOCS →</a>
    <a href="/tasks">TASKS →</a>
    <a href="/health">HEALTH →</a>
  </div>
</main>
<footer>GridWorld Survival · OpenEnv-compliant · Built for AI agent evaluation</footer>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)