"""
Local AI Stack — FastAPI + WebSocket backend
Run: python ~/agents/app.py
Access: http://localhost:7860
"""
import os, sys, json, re, subprocess, threading, requests, signal, atexit
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from crewai import Agent, Task, Crew, LLM

# ── Paths ─────────────────────────────────────────────────
AGENTS_DIR   = "/home/ilyes/agents"
LOG_DIR      = os.path.join(AGENTS_DIR, "logs")
MEMORY_FILE  = os.path.join(AGENTS_DIR, "memory.json")
PROFILE_FILE = os.path.join(AGENTS_DIR, "profile.json")
GRAPH_FILE   = os.path.expanduser("~/Pictures/graph.png")
os.makedirs(LOG_DIR, exist_ok=True)
sys.path.insert(0, AGENTS_DIR)

# ── Models ────────────────────────────────────────────────
router_llm  = LLM(model="ollama/qwen2.5:3b",       base_url="http://localhost:11434")
coder_llm   = LLM(model="ollama/qwen2.5-coder:7b", base_url="http://localhost:11434")
math_llm    = LLM(model="ollama/deepseek-r1:7b",    base_url="http://localhost:11434")
general_llm = LLM(model="ollama/qwen2.5:7b",        base_url="http://localhost:11434")

# ── Helpers ───────────────────────────────────────────────
def load_profile():
    try:
        p = json.load(open(PROFILE_FILE))
        return "User profile:\n" + "\n".join(f"- {k}: {v}" for k,v in p.items())
    except: return ""

def load_memory():
    try: return json.load(open(MEMORY_FILE))
    except: return []

def save_memory(mem):
    json.dump(mem[-20:], open(MEMORY_FILE,"w"), indent=2)

def memory_ctx(mem):
    if not mem: return ""
    lines = ["Recent history:"]
    for m in mem[-6:]:
        lines += [f"User: {m['q']}", f"Assistant: {m['a'][:200]}"]
    return "\n".join(lines)

def log_it(q, a, cat):
    f = os.path.join(LOG_DIR, datetime.now().strftime("%Y-%m-%d") + ".log")
    open(f,"a").write(f"\n[{datetime.now().strftime('%H:%M:%S')}][{cat}]\nYou: {q}\nA: {a}\n{'-'*50}\n")

def clean(text):
    for p in [r'This is the expected criteria.*?(?=\n\n|\Z)',
              r'you MUST return.*?(?=\n|\Z)',
              r'Provide your complete response:\s*',
              r'### (System|User|Assistant):\s*',
              r'Current Task:.*?(?=\n\n|\Z)',
              r'Your personal goal is:.*?(?=\n|\Z)',
              r'Atentamente,.*?(?=\Z)']:
        text = re.sub(p, '', text, flags=re.DOTALL|re.IGNORECASE)
    for _ in range(5):
        text = re.sub(r'```(\w*)\n(.*?)\n```\s*\n\s*```(\w*)\n',
                      lambda m: f'```{m.group(1) or m.group(3)}\n{m.group(2)}\n',
                      text, flags=re.DOTALL)
    return re.sub(r'\n{3,}','\n\n', text).strip()

def run_crew(agent, desc):
    task = Task(description=desc, expected_output="A helpful response.", agent=agent)
    return clean(str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip())

def search(q):
    try:
        r = requests.get("http://localhost:8888/search",
                         params={"q":q,"format":"json"}, timeout=10)
        res = r.json().get("results",[])[:5]
        return "\n\n".join(f"{i+1}. {r.get('title','')}\n{r.get('content','')}"
                           for i,r in enumerate(res)) or "No results."
    except Exception as e: return f"Search error: {e}"

def read_file(path):
    path = os.path.expanduser(path.strip())
    if os.path.isdir(path):
        items = [f"  {'[DIR]' if os.path.isdir(os.path.join(path,i)) else '[FILE]'} {i}"
                 for i in sorted(os.listdir(path))]
        listing = f"Directory: {path}\n" + "\n".join(items)
        texts = []
        for i in os.listdir(path):
            full = os.path.join(path,i)
            if os.path.isfile(full) and i.endswith((".py",".txt",".md",".c",".cpp",".js")):
                try: texts.append(f"=== {i} ===\n{open(full,errors='ignore').read()[:2000]}")
                except: pass
        return (listing + "\n\n" + "\n\n".join(texts[:3]))[:8000]
    try:
        if path.endswith(".pdf"):
            return subprocess.run(["pdftotext",path,"-"],capture_output=True,text=True).stdout[:8000]
        return open(path,errors="ignore").read()[:8000]
    except Exception as e: return f"Error: {e}"

# ── Agents ────────────────────────────────────────────────
router = Agent(role="Router", goal="Classify request.",
    backstory=(
        "Output ONLY one word: CODE, MATH, SEARCH, TTS, READ, FIXTEXT, ANALYZE, FILE, AIDER, or GENERAL.\n"
        "CODE=programming. MATH=integrals/equations/algebra (NOT simple arithmetic).\n"
        "SEARCH=current news/versions/prices. TTS=speak text user wrote.\n"
        "READ=fetch external content to read aloud. FIXTEXT=fix grammar.\n"
        "ANALYZE=analyze file/folder. FILE=create/convert files.\n"
        "AIDER=work on code project. GENERAL=everything else including stable facts, chat, simple math."
    ), llm=router_llm, verbose=False)

coder   = Agent(role="Coder",    goal="Write, explain, debug code.",
    backstory="Expert coder. Show code first, explain after. Never expose internal formatting.",
    llm=coder_llm)
math_ag = Agent(role="Math",     goal="Solve math step by step.",
    backstory="Expert mathematician. Number steps, show all work, state method and final answer.",
    llm=math_llm)
general = Agent(role="Assistant", goal="Answer questions concisely.",
    backstory=(
        "You are Local AI Stack on the user's machine. Use profile and history. "
        "Be concise, direct, honest. Never expose internal CrewAI formatting."
    ), llm=general_llm)
reader  = Agent(role="Reader", goal="Extract clean text to speak.",
    backstory="Output ONLY raw text to be spoken. No URLs, metadata, commentary.",
    llm=general_llm)
fixer   = Agent(role="Fixer",  goal="Fix grammar and punctuation.",
    backstory="Fix errors, output ONLY corrected text.", llm=general_llm, verbose=False)
analyzer= Agent(role="Analyst", goal="Analyze file/folder content.",
    backstory="Analyze thoroughly. Never expose internal formatting.", llm=general_llm)

# ── Router ────────────────────────────────────────────────
GENERAL_KW = ["capital of","what is the capital","definition of","what is a ","what is an ",
               "my name","who am i","how are you","what can you"]
SEARCH_KW  = ["latest","current version","today's news","breaking","price of","weather in","who won"]

def route(text):
    t = text.lower()
    if any(w in t for w in ["integral","derivative","primitive","antiderivative","equation",
                              "solve for","matrix","eigenvalue","differential equation"]): return "MATH"
    if any(w in t for w in ["read me the","recite the","read aloud the"]): return "READ"
    if any(w in t for w in ["fix my text","correct this text","fix grammar","fix punctuation"]): return "FIXTEXT"
    if any(w in t for w in ["analyze ","summarize this file","review this file","what does this file"]): return "ANALYZE"
    if any(w in t for w in ["open aider","work on my project","refactor my","edit my code"]): return "AIDER"
    if any(w in t for w in ["create a file","make a file","write a file","convert to pdf","export to"]): return "FILE"
    if any(w in t for w in GENERAL_KW): return "GENERAL"
    if any(w in t for w in SEARCH_KW): return "SEARCH"
    task = Task(description=f"Classify: '{text}'",
                expected_output="One word: CODE MATH SEARCH TTS READ FIXTEXT ANALYZE FILE AIDER GENERAL",
                agent=router)
    res = str(Crew(agents=[router],tasks=[task],verbose=False).kickoff()).strip().upper()
    for kw in ["CODE","MATH","SEARCH","READ","TTS","FIXTEXT","ANALYZE","FILE","AIDER","GENERAL"]:
        if kw in res: return kw
    return "GENERAL"

def speak_bg(text):
    try:
        from tts import speak_auto
        speak_auto(text, "en", None, 1.1)
    except Exception as e: print(f"[TTS] {e}")

# ── Process message ───────────────────────────────────────
def process_message(message, memory_store):
    cat     = route(message)
    profile = load_profile()
    ctx     = memory_ctx(memory_store)
    result  = ""

    if cat == "CODE":
        result = run_crew(coder, f"{ctx}\n\nCoding request: {message}" if ctx else f"Coding request: {message}")

    elif cat == "MATH":
        result = run_crew(math_ag,
            f"Solve step by step: {message}\nState method, number steps, show all work, state final answer.")

    elif cat == "SEARCH":
        result = run_crew(general,
            f"Answer directly and concisely using these search results. No greetings.\n\n"
            f"Results:\n{search(message)}\n\nQuestion: {message}")

    elif cat == "TTS":
        quoted = re.findall(r'"([^"]+)"', message)
        txt = " ".join(quoted) if quoted else message
        for w in ["say","speak","read this","read aloud"]: txt = txt.replace(w,"").strip()
        threading.Thread(target=speak_bg, args=(txt,), daemon=True).start()
        result = f"🔊 Speaking: *{txt}*"
        log_it(message, result, cat)
        return cat, result, memory_store

    elif cat == "READ":
        txt = run_crew(reader,
            f"Extract ONLY the raw spoken text for: '{message}'\n"
            f"Results:\n{search(message)}\nOutput ONLY the text to be spoken.")
        threading.Thread(target=speak_bg, args=(txt,), daemon=True).start()
        result = f"🔊 Reading ({len(txt.split())} words):\n\n{txt}"
        log_it(message, result, cat)
        return cat, result, memory_store

    elif cat == "FIXTEXT":
        fix = message
        for p in ["fix my text","correct this","fix grammar","fix punctuation","fix"]:
            fix = fix.replace(p,"").strip()
        result = run_crew(fixer, f"Fix: '{fix}'")

    elif cat == "ANALYZE":
        words = message.split()
        fp = None
        for w in words:
            expanded = os.path.expanduser(w)
            if os.path.exists(expanded):
                fp = expanded
                break
        if not fp:
            for i, w in enumerate(words):
                if w.startswith("~") or w.startswith("/"):
                    for j in range(len(words), i, -1):
                        candidate = os.path.expanduser(" ".join(words[i:j]))
                        if os.path.exists(candidate):
                            fp = candidate
                            break
                if fp: break
        if not fp:
            return cat, "❌ Could not find that path. Try the full path like `~/Documents/...`", memory_store
        q = message
        for w in words:
            if os.path.exists(os.path.expanduser(w)): q = q.replace(w,"").strip()
        for cmd in ["analyze","summarize","review","what does","read"]:
            q = q.replace(cmd,"").strip()
        result = run_crew(analyzer, f"Content:\n{read_file(fp)}\n\nRequest: {q or 'Summarize.'}")

    elif cat == "AIDER":
        words = message.split()
        project = next((os.path.expanduser(w) for w in words
                        if os.path.exists(os.path.expanduser(w)) and os.path.isdir(os.path.expanduser(w))),
                       os.path.expanduser("~/agents"))
        subprocess.Popen([
            "kitty", "-e", "bash", "-c",
            f"source /home/ilyes/.venvs/aider/bin/activate && "
            f"cd {project} && aider --model ollama/qwen2.5-coder:7b; read"
        ])
        result = f"🛠️ Launched Aider in `{project}` — check your new Kitty window"
        log_it(message, result, cat)
        return cat, result, memory_store

    elif cat == "FILE":
        result = "⚠️ FILE creation requires the terminal. Run: `python ~/agents/main.py`"
        return cat, result, memory_store

    else:  # GENERAL
        parts = [p for p in [profile, ctx] if p]
        parts.append(
            f"Answer directly and concisely. No greetings, no sign-offs. "
            f"Never expose internal formatting. Question: {message}"
        )
        result = run_crew(general, "\n\n".join(parts))

    log_it(message, result, cat)
    memory_store = memory_store + [{"q": message, "a": result, "cat": cat}]
    save_memory(memory_store)
    return cat, result, memory_store

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/memory")
def api_memory():
    return {"memory": load_memory()[-8:]}

@app.get("/api/profile")
def api_profile():
    try: return json.load(open(PROFILE_FILE))
    except: return {}

@app.get("/api/logs")
def api_logs():
    if not os.path.exists(LOG_DIR): return {"content": ""}
    files = sorted(os.listdir(LOG_DIR), reverse=True)
    if not files: return {"content": ""}
    try:
        content = open(os.path.join(LOG_DIR, files[0])).read()
        return {"filename": files[0], "content": content}
    except: return {"content": ""}

@app.post("/api/clear-memory")
def api_clear_memory():
    save_memory([])
    return {"ok": True}

@app.post("/api/stop")
def api_stop():
    def _stop():
        import time; time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=_stop, daemon=True).start()
    return {"ok": True}

# ── WebSocket for chat ────────────────────────────────────
memory_store = load_memory()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global memory_store
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "").strip()
            if not message:
                continue
            await websocket.send_json({"type": "thinking", "category": "..."})
            try:
                cat, result, memory_store = process_message(message, memory_store)
                await websocket.send_json({
                    "type": "response",
                    "category": cat,
                    "content": result,
                    "memory": [{"q": m["q"][:60], "cat": m.get("cat","?")} for m in memory_store[-8:]]
                })
            except Exception as e:
                await websocket.send_json({"type": "error", "content": str(e)})
    except WebSocketDisconnect:
        pass

# ── Serve the HTML frontend ───────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>Local AI Stack</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#070a0f;--surface:#0d1117;--card:#0c1220;--border:#1e2d3d;
  --accent:#00d4ff;--text:#e2e8f0;--muted:#5a6a8a;--danger:#ef4444;
  --green:#10b981;--amber:#f59e0b;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:'Courier New',monospace;font-size:14px;overflow:hidden}
body{display:flex;flex-direction:column}

/* Header */
.header{
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 20px;background:var(--surface);border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.logo{display:flex;align-items:center;gap:8px}
.logo-text{font-size:16px;font-weight:700;letter-spacing:.05em}
.logo-text span{color:var(--accent)}
.header-links{display:flex;gap:8px}
.header-link{
  font-size:11px;color:var(--muted);text-decoration:none;
  border:1px solid var(--border);padding:3px 8px;border-radius:4px;
  transition:color .2s,border-color .2s;
}
.header-link:hover{color:var(--accent);border-color:var(--accent)}

/* Main layout */
.main{display:flex;flex:1;overflow:hidden}

/* Sidebar */
.sidebar{
  width:180px;border-right:1px solid var(--border);
  display:flex;flex-direction:column;flex-shrink:0;overflow-y:auto;
}
.sidebar-section{padding:12px}
.sidebar-label{font-size:9px;letter-spacing:.3em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.route-item{font-size:11px;color:var(--muted);padding:4px 6px;border-radius:4px;display:flex;align-items:center;gap:6px}
.route-item span{color:var(--accent)}
.sidebar-tabs{display:flex;border-bottom:1px solid var(--border)}
.sidebar-tab{
  flex:1;padding:8px;font-size:10px;letter-spacing:.1em;text-transform:uppercase;
  background:transparent;border:none;color:var(--muted);cursor:pointer;
  border-bottom:2px solid transparent;transition:all .2s;font-family:inherit;
}
.sidebar-tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.sidebar-panel{padding:12px;flex:1;overflow-y:auto;display:none}
.sidebar-panel.active{display:block}
.mem-item{font-size:11px;color:var(--muted);padding:6px 0;border-bottom:1px solid var(--border);line-height:1.4}
.mem-cat{color:var(--accent);font-size:10px}
.profile-item{font-size:11px;padding:4px 0;border-bottom:1px solid var(--border)}
.profile-key{color:var(--accent)}
.log-content{font-size:10px;color:var(--muted);white-space:pre-wrap;line-height:1.5}
.sidebar-btn{
  width:100%;padding:7px;margin-top:8px;background:transparent;
  border:1px solid var(--border);color:var(--muted);border-radius:5px;
  font-size:11px;cursor:pointer;font-family:inherit;transition:all .2s;
}
.sidebar-btn:hover{border-color:var(--accent);color:var(--accent)}
.sidebar-btn.danger:hover{border-color:var(--danger);color:var(--danger)}

/* Chat area */
.chat-area{flex:1;display:flex;flex-direction:column;overflow:hidden}
.messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.message{max-width:85%;animation:fadeIn .2s ease}
.message.user{align-self:flex-end}
.message.assistant{align-self:flex-start}
.msg-bubble{
  padding:10px 14px;border-radius:10px;line-height:1.6;
  word-break:break-word;white-space:pre-wrap;
}
.message.user .msg-bubble{
  background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.2);
  border-radius:10px 10px 2px 10px;
}
.message.assistant .msg-bubble{
  background:var(--surface);border:1px solid var(--border);
  border-radius:10px 10px 10px 2px;
}
.msg-cat{font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);margin-bottom:4px}
.msg-time{font-size:10px;color:var(--muted);margin-top:4px;text-align:right}

/* Code blocks */
.msg-bubble pre{
  background:#050810;border:1px solid var(--border);border-radius:6px;
  padding:10px;margin:8px 0;overflow-x:auto;font-size:12px;
}
.msg-bubble code{color:#a8c4e8;font-family:'Courier New',monospace}
.msg-bubble p{margin-bottom:6px}
.msg-bubble p:last-child{margin-bottom:0}
.msg-bubble strong{color:var(--text)}

/* Thinking indicator */
.thinking .msg-bubble{color:var(--muted);font-style:italic}
.dot-anim::after{content:'...';animation:dots 1.2s infinite}
@keyframes dots{0%{content:'.'}33%{content:'..'}66%{content:'...'}}

/* Input area */
.input-area{
  padding:12px 16px;border-top:1px solid var(--border);
  background:var(--surface);display:flex;gap:8px;align-items:flex-end;flex-shrink:0;
}
.input-wrap{flex:1;position:relative}
textarea{
  width:100%;background:var(--card);border:1px solid var(--border);
  color:var(--text);padding:10px 14px;border-radius:8px;resize:none;
  font-family:'Courier New',monospace;font-size:14px;line-height:1.5;
  max-height:120px;transition:border-color .2s;outline:none;
}
textarea:focus{border-color:var(--accent)}
.send-btn{
  background:var(--accent);color:#000;border:none;border-radius:8px;
  padding:10px 18px;font-weight:700;cursor:pointer;font-family:inherit;
  font-size:13px;transition:opacity .2s;white-space:nowrap;flex-shrink:0;
}
.send-btn:hover{opacity:.9}
.send-btn:disabled{opacity:.5;cursor:not-allowed}

/* Status bar */
.status-bar{
  padding:4px 16px;background:var(--bg);border-top:1px solid var(--border);
  font-size:10px;color:var(--muted);display:flex;justify-content:space-between;
  flex-shrink:0;
}
.status-dot{width:6px;height:6px;border-radius:50%;background:var(--green);display:inline-block;margin-right:5px}
.status-dot.offline{background:var(--danger)}

/* Mobile */
@media(max-width:768px){
  .sidebar{display:none}
  .header-links{display:none}
  .chat-area{width:100%}
  .messages{padding:10px}
  .input-area{padding:8px 10px}
  textarea{font-size:16px}
  .msg-bubble{font-size:13px}
  .mobile-menu-btn{display:flex !important}
}
.mobile-menu-btn{
  display:none;background:transparent;border:1px solid var(--border);
  color:var(--muted);padding:5px 8px;border-radius:4px;cursor:pointer;
  font-size:14px;font-family:inherit;
}

/* Mobile sidebar overlay */
.sidebar-overlay{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:10;
}
.sidebar-overlay.open{display:block}
.sidebar.mobile-open{
  display:flex;position:fixed;left:0;top:0;bottom:0;z-index:11;
  background:var(--surface);box-shadow:4px 0 20px rgba(0,0,0,.5);
}

@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}

/* Scrollbar */
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

/* Markdown in messages */
.msg-bubble h1,.msg-bubble h2,.msg-bubble h3{color:var(--accent);margin:8px 0 4px}
.msg-bubble ul,.msg-bubble ol{padding-left:16px;margin:4px 0}
.msg-bubble li{margin:2px 0}
.msg-bubble hr{border:none;border-top:1px solid var(--border);margin:8px 0}
</style>
</head>
<body>

<div class="header">
  <div class="logo">
    <button class="mobile-menu-btn" onclick="toggleSidebar()">☰</button>
    <div class="logo-text">Local AI <span>Stack</span></div>
  </div>
  <div class="header-links">
    <a class="header-link" href="http://localhost:8080" target="_blank">WebUI</a>
    <a class="header-link" href="http://localhost:8888" target="_blank">SearXNG</a>
    <a class="header-link" href="http://localhost:8188" target="_blank">ComfyUI</a>
  </div>
</div>

<div class="main">
  <div class="sidebar" id="sidebar">
    <div class="sidebar-section">
      <div class="sidebar-label">Routes</div>
      <div id="routes-list"></div>
    </div>

    <div class="sidebar-tabs">
      <button class="sidebar-tab active" onclick="showPanel('memory',this)">Mem</button>
      <button class="sidebar-tab" onclick="showPanel('profile',this)">Profile</button>
      <button class="sidebar-tab" onclick="showPanel('logs',this)">Logs</button>
      <button class="sidebar-tab" onclick="showPanel('server',this)">Server</button>
    </div>

    <div class="sidebar-panel active" id="panel-memory">
      <div id="memory-list"><div style="color:var(--muted);font-size:11px">No memory yet.</div></div>
      <button class="sidebar-btn danger" onclick="clearMemory()">Clear Memory</button>
    </div>

    <div class="sidebar-panel" id="panel-profile">
      <div id="profile-content"><div style="color:var(--muted);font-size:11px">Loading...</div></div>
    </div>

    <div class="sidebar-panel" id="panel-logs">
      <div id="logs-content"><div style="color:var(--muted);font-size:11px">Loading...</div></div>
      <button class="sidebar-btn" onclick="loadLogs()">Refresh</button>
    </div>

    <div class="sidebar-panel" id="panel-server">
      <div id="server-status" style="font-size:11px;color:var(--muted);margin-bottom:10px;">
        🔒 Private — localhost:7860
      </div>
      <button class="sidebar-btn" onclick="makePublic()">🌐 Make Public</button>
      <button class="sidebar-btn danger" onclick="stopServer()">⏹ Stop Server</button>
    </div>
  </div>

  <div class="sidebar-overlay" id="overlay" onclick="toggleSidebar()"></div>

  <div class="chat-area">
    <div class="messages" id="messages">
      <div class="message assistant">
        <div class="msg-cat">SYSTEM</div>
        <div class="msg-bubble">🤖 Local AI Stack ready. Type a message to get started.
Routes: CODE · MATH · SEARCH · TTS · READ · FIXTEXT · ANALYZE · AIDER · GENERAL</div>
      </div>
    </div>
    <div class="input-area">
      <div class="input-wrap">
        <textarea id="input" placeholder="Ask anything..." rows="1"
          onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
      </div>
      <button class="send-btn" id="send-btn" onclick="sendMessage()">Send</button>
    </div>
  </div>
</div>

<div class="status-bar">
  <div><span class="status-dot" id="status-dot"></span><span id="status-text">Connected</span></div>
  <div id="status-right">localhost:7860</div>
</div>

<script>
// ── WebSocket ─────────────────────────────────────────────
const routes = ["CODE","MATH","SEARCH","TTS","READ","FIXTEXT","ANALYZE","FILE","AIDER","GENERAL"];
let ws, reconnectTimer;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen = () => {
    setStatus(true);
    clearTimeout(reconnectTimer);
  };

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === 'thinking') {
      showThinking(data.category);
    } else if (data.type === 'response') {
      removeThinking();
      appendMessage('assistant', data.category, data.content);
      if (data.memory) updateMemory(data.memory);
      setSendEnabled(true);
    } else if (data.type === 'error') {
      removeThinking();
      appendMessage('assistant', 'ERROR', '❌ ' + data.content);
      setSendEnabled(true);
    }
  };

  ws.onclose = () => {
    setStatus(false);
    reconnectTimer = setTimeout(connect, 3000);
  };

  ws.onerror = () => ws.close();
}

function setStatus(ok) {
  document.getElementById('status-dot').className = 'status-dot' + (ok ? '' : ' offline');
  document.getElementById('status-text').textContent = ok ? 'Connected' : 'Reconnecting...';
}

// ── Routes sidebar ────────────────────────────────────────
const routeList = document.getElementById('routes-list');
routes.forEach(r => {
  const el = document.createElement('div');
  el.className = 'route-item';
  el.innerHTML = `<span>→</span>${r}`;
  routeList.appendChild(el);
});

// ── Chat ──────────────────────────────────────────────────
function sendMessage() {
  const input = document.getElementById('input');
  const msg = input.value.trim();
  if (!msg || !ws || ws.readyState !== WebSocket.OPEN) return;
  appendMessage('user', null, msg);
  setSendEnabled(false);
  ws.send(JSON.stringify({message: msg}));
  input.value = '';
  input.style.height = 'auto';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function setSendEnabled(on) {
  document.getElementById('send-btn').disabled = !on;
}

let thinkingEl = null;
function showThinking(cat) {
  removeThinking();
  thinkingEl = document.createElement('div');
  thinkingEl.className = 'message assistant thinking';
  thinkingEl.innerHTML = `<div class="msg-cat">${cat}</div><div class="msg-bubble"><span class="dot-anim">Thinking</span></div>`;
  document.getElementById('messages').appendChild(thinkingEl);
  scrollBottom();
}

function removeThinking() {
  if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
}

function appendMessage(role, cat, content) {
  const msgs = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = `message ${role}`;
  const time = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  el.innerHTML = `
    ${cat ? `<div class="msg-cat">${cat}</div>` : ''}
    <div class="msg-bubble">${renderMarkdown(content)}</div>
    <div class="msg-time">${time}</div>
  `;
  msgs.appendChild(el);
  scrollBottom();
}

function scrollBottom() {
  const msgs = document.getElementById('messages');
  msgs.scrollTop = msgs.scrollHeight;
}

// ── Simple markdown renderer ──────────────────────────────
function renderMarkdown(text) {
  // Escape HTML first
  let h = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  // Code blocks
  h = h.replace(/```(\w*)\n([\s\S]*?)```/g, (_,lang,code) =>
    `<pre><code>${code}</code></pre>`);
  // Inline code
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  h = h.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // Italic
  h = h.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  // Headers
  h = h.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  h = h.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  h = h.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  // Bullet lists
  h = h.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
  h = h.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
  // Numbered lists
  h = h.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  // Line breaks
  h = h.replace(/\n\n/g, '</p><p>');
  h = h.replace(/\n/g, '<br>');
  return h;
}

// ── Sidebar panels ────────────────────────────────────────
function showPanel(name, btn) {
  document.querySelectorAll('.sidebar-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.sidebar-tab').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'profile') loadProfile();
  if (name === 'logs') loadLogs();
}

function updateMemory(mem) {
  const el = document.getElementById('memory-list');
  if (!mem.length) { el.innerHTML = '<div style="color:var(--muted);font-size:11px">No memory yet.</div>'; return; }
  el.innerHTML = mem.map(m =>
    `<div class="mem-item"><span class="mem-cat">[${m.cat}]</span> ${m.q}...</div>`
  ).join('');
}

async function loadProfile() {
  const res = await fetch('/api/profile');
  const data = await res.json();
  const el = document.getElementById('profile-content');
  if (!Object.keys(data).length) {
    el.innerHTML = '<div style="color:var(--muted);font-size:11px">No profile. Create ~/agents/profile.json</div>';
    return;
  }
  el.innerHTML = Object.entries(data).map(([k,v]) =>
    `<div class="profile-item"><span class="profile-key">${k}:</span> ${v}</div>`
  ).join('');
}

async function loadLogs() {
  const res = await fetch('/api/logs');
  const data = await res.json();
  const el = document.getElementById('logs-content');
  if (!data.content) { el.innerHTML = '<div style="color:var(--muted);font-size:11px">No logs yet.</div>'; return; }
  const lines = data.content.trim().split('\n').slice(-40);
  el.innerHTML = `<div style="font-size:10px;color:var(--muted);margin-bottom:6px">${data.filename || ''}</div>
    <div class="log-content">${lines.join('\n')}</div>`;
  el.scrollTop = el.scrollHeight;
}

async function clearMemory() {
  await fetch('/api/clear-memory', {method:'POST'});
  updateMemory([]);
}

function makePublic() {
  document.getElementById('server-status').innerHTML =
    `🌐 <strong>To make public:</strong><br><br>
    Restart with:<br>
    <code style="font-size:10px;color:var(--accent)">GRADIO_SHARE=true python ~/agents/app.py</code><br><br>
    <span style="color:var(--muted);font-size:10px">⚠️ Anyone with the link can access your AI</span>`;
}

async function stopServer() {
  if (!confirm('Stop the server?')) return;
  await fetch('/api/stop', {method:'POST'});
  document.getElementById('server-status').textContent = '⏹ Server stopped.';
  setStatus(false);
}

// ── Mobile sidebar ────────────────────────────────────────
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('overlay');
  sidebar.classList.toggle('mobile-open');
  overlay.classList.toggle('open');
}

// ── Init ──────────────────────────────────────────────────
connect();
loadProfile();
loadLogs();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML

# ── Cleanup ───────────────────────────────────────────────
def cleanup(signum=None, frame=None):
    subprocess.run(["fuser", "-k", "7860/tcp"], capture_output=True)
    os._exit(0)

signal.signal(signal.SIGINT,  cleanup)
signal.signal(signal.SIGTERM, cleanup)
atexit.register(cleanup)

if __name__ == "__main__":
    subprocess.run(["fuser", "-k", "7860/tcp"], capture_output=True)
    import time; time.sleep(0.5)
    print("🤖 Local AI Stack — Web Interface")
    print("   Local:  http://localhost:7860")
    print("   Mobile: http://192.168.0.20:7860\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
