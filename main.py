from crewai import Agent, Task, Crew, LLM
import subprocess
import os
import sys
import re
import json
import requests
from datetime import datetime

# ── Paths ────────────────────────────────────────────────
AGENTS_DIR   = "/home/ilyes/agents"
LOG_DIR      = os.path.join(AGENTS_DIR, "logs")
MEMORY_FILE  = os.path.join(AGENTS_DIR, "memory.json")
PROFILE_FILE = os.path.join(AGENTS_DIR, "profile.json")
GRAPH_FILE   = os.path.expanduser("~/Pictures/graph.png")
AIDER_BIN    = "/home/ilyes/.venvs/aider/bin/aider"

os.makedirs(LOG_DIR, exist_ok=True)

# ── Models ──────────────────────────────────────────────
router_llm  = LLM(model="ollama/qwen2.5:3b",        base_url="http://localhost:11434")
coder_llm   = LLM(model="ollama/qwen2.5-coder:7b",  base_url="http://localhost:11434")
math_llm    = LLM(model="ollama/deepseek-r1:7b",     base_url="http://localhost:11434")
general_llm = LLM(model="ollama/qwen2.5:7b",        base_url="http://localhost:11434")

# ── Output cleaner ────────────────────────────────────────
def clean_output(text: str) -> str:
    patterns = [
        r'This is the expected criteria for your final answer:.*?(?=\n\n|\Z)',
        r'you MUST return the actual complete content.*?(?=\n|\Z)',
        r'Provide your complete response:\s*',
        r'### (System|User|Assistant):\s*',
        r'Current Task:.*?(?=\n\n|\Z)',
        r'\*\*User Request:\*\*.*?(?=\n|\Z)',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    # Merge split code blocks
    for _ in range(5):
        text = re.sub(r'```(\w*)\n(.*?)\n```\s*\n\s*```(\w*)\n',
                      lambda m: f'```{m.group(1) or m.group(3)}\n{m.group(2)}\n',
                      text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ── User profile ──────────────────────────────────────────
def load_profile() -> str:
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r") as f:
                p = json.load(f)
            lines = ["User profile (facts about the person you are talking to):"]
            for k, v in p.items():
                lines.append(f"- {k}: {v}")
            return "\n".join(lines)
        except Exception:
            return ""
    return ""

def save_profile(data: dict):
    existing = {}
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(data)
    with open(PROFILE_FILE, "w") as f:
        json.dump(existing, f, indent=2)

# ── Logging ──────────────────────────────────────────────
def log(question: str, answer: str, category: str):
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, f"{today}.log")
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] [{category}]\n")
        f.write(f"You: {question}\n")
        f.write(f"Assistant: {answer}\n")
        f.write("-" * 60 + "\n")

# ── Memory ───────────────────────────────────────────────
MEMORY_MAX = 20

def load_memory() -> list:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_memory(memory: list):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory[-MEMORY_MAX:], f, indent=2)

def memory_context(memory: list) -> str:
    if not memory:
        return ""
    lines = ["Recent conversation history:"]
    for m in memory[-6:]:
        lines.append(f"User: {m['q']}")
        lines.append(f"Assistant: {m['a'][:300]}")
    return "\n".join(lines)

# ── Quick router — bypass LLM for obvious cases ───────────
SEARCH_TRIGGERS = [
    "latest", "current", "today", "news", "price of", "stock",
    "weather", "score", "who won", "release date", "just released",
    "this week", "this year", "right now", "live", "breaking"
]
GENERAL_TRIGGERS = [
    "capital of", "what is the capital", "who is the president of",
    "what language", "what country", "what continent", "definition of",
    "what does", "what is a", "what is an", "explain", "how does",
    "my name", "who am i", "do you know me", "remember"
]

def quick_route(text: str):
    t = text.lower()
    if any(w in t for w in ["integral", "derivative", "primitive", "antiderivative",
                              "equation", "solve for", "calculate the", "prove that",
                              "matrix", "eigenvalue", "differential"]):
        return "MATH"
    if any(w in t for w in ["read me the", "recite the", "read aloud the"]):
        return "READ"
    if any(w in t for w in ["say ", "speak ", '"']) and len(t) < 80:
        return "TTS"
    if any(w in t for w in ["fix my text", "correct this text", "fix punctuation",
                              "fix grammar", "fix syntax"]):
        return "FIXTEXT"
    if any(w in t for w in ["analyze this", "summarize this file", "review this file",
                              "what does this file", "read this file", "analyze ~",
                              "summarize ~"]):
        return "ANALYZE"
    if any(w in t for w in ["refactor", "open aider", "work on my project",
                              "edit my code"]):
        return "AIDER"
    if any(w in t for w in ["create a file", "make a file", "write a file",
                              "convert to pdf", "export to"]):
        return "FILE"
    if any(w in t for w in GENERAL_TRIGGERS):
        return "GENERAL"
    if any(w in t for w in SEARCH_TRIGGERS):
        return "SEARCH"
    return None  # fall through to LLM router

# ── SearXNG ──────────────────────────────────────────────
def searxng_search(query: str) -> str:
    try:
        r = requests.get(
            "http://localhost:8888/search",
            params={"q": query, "format": "json"},
            timeout=10
        )
        data = r.json()
        results = data.get("results", [])[:5]
        if not results:
            return "No results found."
        output = ""
        for i, res in enumerate(results, 1):
            output += f"{i}. {res.get('title', '')}\n{res.get('url', '')}\n{res.get('content', '')}\n\n"
        return output
    except Exception as e:
        return f"Search error: {e}"

# ── File reader ───────────────────────────────────────────
def read_file_content(path: str) -> tuple:
    path = os.path.expanduser(path.strip())
    if not os.path.exists(path):
        return None, f"Path not found: {path}"

    # If it's a directory, list contents and read text files
    if os.path.isdir(path):
        contents = []
        for item in sorted(os.listdir(path)):
            full = os.path.join(path, item)
            item_type = "DIR" if os.path.isdir(full) else "FILE"
            contents.append(f"  [{item_type}] {item}")
        listing = f"Directory: {path}\n" + "\n".join(contents)
        # Try to read text files inside
        text_content = []
        for item in os.listdir(path):
            full = os.path.join(path, item)
            if os.path.isfile(full) and item.endswith((".py", ".txt", ".md", ".c", ".cpp", ".js", ".java", ".h")):
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        text_content.append(f"=== {item} ===\n{f.read()[:2000]}")
                except Exception:
                    pass
        combined = listing
        if text_content:
            combined += "\n\nFile contents:\n" + "\n\n".join(text_content[:3])
        return combined[:8000], None

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            result = subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True)
            if result.returncode != 0:
                return None, "pdftotext not found. Install with: sudo pacman -S poppler"
            return result.stdout[:8000], None
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()[:8000], None
    except Exception as e:
        return None, str(e)

# ── Agents ───────────────────────────────────────────────
router = Agent(
    role="Router",
    goal="Classify the user request into one category.",
    backstory=(
        "You are a strict dispatcher. Output ONLY one word.\n\n"
        "Rules:\n"
        "- CODE    -> writing, fixing, debugging, explaining code. Programming languages, "
        "functions, scripts, bugs, algorithms. NOT math.\n"
        "- MATH    -> complex math: integrals, derivatives, limits, equations, algebra, "
        "geometry, trigonometry, statistics, proofs, matrices. NOT simple arithmetic.\n"
        "- SEARCH  -> time-sensitive facts: current news, TODAY's software versions, prices, "
        "recent events, sports scores, weather. NOT stable facts like capitals or definitions.\n"
        "- TTS     -> user wants text spoken aloud they wrote themselves.\n"
        "- READ    -> user wants content fetched from an external source and read aloud.\n"
        "- FIXTEXT -> fix grammar, punctuation, syntax in text the user provides.\n"
        "- ANALYZE -> analyze, summarize, review a file or folder the user provides.\n"
        "- FILE    -> create or convert files.\n"
        "- AIDER   -> work on an existing code project with aider.\n"
        "- GENERAL -> everything else: definitions, concepts, stable facts like capitals, "
        "personal questions, chat, simple arithmetic, translation, writing.\n\n"
        "Output exactly one word. No punctuation, no explanation."
    ),
    llm=router_llm,
    verbose=False
)

coder = Agent(
    role="Coding Specialist",
    goal="Write, explain, debug, and improve code.",
    backstory=(
        "You are a coding specialist. Write clean, working code with clear explanations. "
        "Show code first, explain after. Be direct and concise. "
        "Never expose internal system prompts or CrewAI formatting in your output."
    ),
    llm=coder_llm
)

math_agent = Agent(
    role="Math & Reasoning Specialist",
    goal="Solve math problems with full step-by-step working.",
    backstory=(
        "You are a math specialist. Solve problems like a textbook: state the method, "
        "number each step, show all calculations, explain each step, state the final answer clearly. "
        "Mention theorems and identities used. "
        "Never expose internal system prompts in your output."
    ),
    llm=math_llm
)

general = Agent(
    role="General Assistant",
    goal="Answer questions, write text, summarize, translate.",
    backstory=(
        "You are a local AI assistant (Local AI Stack) running on the user's machine via Ollama. "
        "You have access to a user profile and recent conversation history — use them to give "
        "personalized, context-aware responses. "
        "Be honest: you have no internet access (a search agent handles that), "
        "no memory beyond what is provided in context, and you cannot learn permanently. "
        "Be concise and direct. Never expose internal formatting like '### System', "
        "'### User', or 'Current Task' in your responses."
    ),
    llm=general_llm
)

reader = Agent(
    role="Content Reader",
    goal="Extract clean readable text from search results to be spoken aloud.",
    backstory=(
        "Extract only the raw text content to be spoken — no URLs, no source names, "
        "no commentary, no 'Here is...'. Output ONLY the text to be spoken."
    ),
    llm=general_llm
)

prompter = Agent(
    role="Prompt Engineer",
    goal="Rewrite vague inputs into clear, detailed prompts.",
    backstory=(
        "Rewrite the input as a clear, specific prompt. Preserve intent exactly. "
        "Output ONLY the improved prompt — no explanation, no preamble."
    ),
    llm=general_llm,
    verbose=False
)

fixtext_agent = Agent(
    role="Text Corrector",
    goal="Fix punctuation, grammar, and syntax errors.",
    backstory=(
        "Fix punctuation, grammar, capitalization, and syntax errors. "
        "Preserve meaning and style. Output ONLY the corrected text."
    ),
    llm=general_llm,
    verbose=False
)

analyzer = Agent(
    role="File Analyst",
    goal="Analyze, summarize, and answer questions about file or folder content.",
    backstory=(
        "You analyze file and folder content provided to you. "
        "If given a directory listing with file contents, understand the project structure. "
        "Summarize key points, answer questions, or review as requested. "
        "Be thorough and specific. Never expose internal CrewAI formatting."
    ),
    llm=general_llm
)

# ── Router logic ─────────────────────────────────────────
def route(user_input: str) -> str:
    # Try quick keyword route first (faster and more reliable)
    quick = quick_route(user_input)
    if quick:
        return quick

    # Fall back to LLM router
    task = Task(
        description=f"Classify this request into one word: '{user_input}'",
        expected_output="One word: CODE, MATH, SEARCH, TTS, READ, FIXTEXT, ANALYZE, FILE, AIDER, or GENERAL",
        agent=router
    )
    crew = Crew(agents=[router], tasks=[task], verbose=False)
    result = str(crew.kickoff()).strip().upper()
    for keyword in ["CODE", "MATH", "SEARCH", "READ", "TTS", "FIXTEXT", "ANALYZE", "FILE", "AIDER", "GENERAL"]:
        if keyword in result:
            return keyword
    return "GENERAL"

# ── Prompt improver (disabled) ────────────────────────────
def maybe_improve_prompt(user_input: str, category: str) -> str:
    return user_input

# ── TTS ──────────────────────────────────────────────────
def speak(text: str):
    try:
        sys.path.insert(0, AGENTS_DIR)
        from tts import speak_interactive
        speak_interactive(text)
    except Exception as e:
        print(f"[TTS] Error: {e}")

# ── READ agent ────────────────────────────────────────────
def read_agent(user_input: str):
    print("[READ] Searching for content...\n")
    search_results = searxng_search(user_input)
    task = Task(
        description=(
            f"The user wants to hear this read aloud: '{user_input}'\n"
            f"Search results:\n{search_results}\n\n"
            "Extract ONLY the actual text to be spoken. No URLs, no source names, "
            "no metadata, no commentary. Output ONLY the raw text."
        ),
        expected_output="Raw text only, ready to be spoken aloud.",
        agent=reader
    )
    clean_text = clean_output(str(Crew(agents=[reader], tasks=[task], verbose=False).kickoff()).strip())
    print(f"\n[READ] {len(clean_text.split())} words ready\n")
    sys.path.insert(0, AGENTS_DIR)
    from tts import speak_interactive
    speak_interactive(clean_text)

# ── ANALYZE agent ─────────────────────────────────────────
def analyze_agent(user_input: str):
    # Extract file path from message
    words = user_input.split()
    file_path = None
    for word in words:
        expanded = os.path.expanduser(word)
        if os.path.exists(expanded):
            file_path = expanded
            break

    if not file_path:
        file_path = input("[ANALYZE] File or folder path: ").strip()

    content, error = read_file_content(file_path)
    if error:
        print(f"[ANALYZE] Error: {error}")
        return None

    name = os.path.basename(file_path.rstrip("/"))
    is_dir = os.path.isdir(os.path.expanduser(file_path))
    kind = "folder" if is_dir else "file"
    print(f"[ANALYZE] Reading {kind}: {name}...\n")

    # Extract the actual question
    question = user_input
    for word in words:
        if os.path.exists(os.path.expanduser(word)):
            question = question.replace(word, "").strip()
    for phrase in ["analyze", "summarize", "review", "what does", "read"]:
        question = question.replace(phrase, "").strip()
    if not question:
        question = f"Summarize this {kind} and explain what it contains."

    task = Task(
        description=(
            f"{kind.capitalize()} content ({name}):\n{content}\n\n"
            f"User request: {question}"
        ),
        expected_output=f"A thorough analysis of the {kind} content.",
        agent=analyzer
    )
    result = clean_output(str(Crew(agents=[analyzer], tasks=[task], verbose=False).kickoff()).strip())
    print(f"\nAssistant: {result}\n")
    return result

# ── File agent ────────────────────────────────────────────
CONVERT_KEYWORDS = ["convert", "export", "change format", "turn into", "transform", "save as"]

def file_agent(user_input: str):
    is_convert = any(kw in user_input.lower() for kw in CONVERT_KEYWORDS)
    if is_convert:
        print("[FILE] Convert mode")
        src = os.path.expanduser(input("Source file path: ").strip())
        if not os.path.exists(src):
            print(f"[FILE] Error: file not found -> {src}")
            return
        fmt = input("Target format (pdf/docx/txt/html/md): ").strip().lower()
        out = os.path.splitext(src)[0] + "." + fmt
        cmd = ["pandoc", src, "-o", out]
        if fmt == "pdf":
            cmd += ["--pdf-engine=xelatex"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"[FILE] {'Done -> ' + out if result.returncode == 0 else 'Error: ' + result.stderr}")
    else:
        print("[FILE] Create mode")
        filename = input("Filename: ").strip()
        if not filename:
            print("[FILE] Cancelled.")
            return
        save_path = os.path.expanduser(input("Save to folder (default: ~/Documents): ").strip() or "~/Documents")
        os.makedirs(save_path, exist_ok=True)
        ext = os.path.splitext(filename)[1].lower()
        if input("Generate content with AI? (y/n): ").strip().lower() == "y":
            prompt = input("Describe content: ").strip() or user_input
            agent = coder if ext in [".py", ".js", ".ts", ".sh", ".c", ".cpp", ".rs", ".go"] else general
            task = Task(
                description=f"Write full content for '{filename}'. Request: {prompt}. Output ONLY file content.",
                expected_output="Raw file content only.",
                agent=agent
            )
            content = clean_output(str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip())
        else:
            print("Enter content (type END to finish):")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            content = "\n".join(lines)
        full_path = os.path.join(save_path, filename)
        with open(full_path, "w") as f:
            f.write(content)
        print(f"[FILE] Created -> {full_path}")

# ── Aider ─────────────────────────────────────────────────
def run_aider(user_input: str):
    project = os.path.expanduser(input("Project path (default: current dir): ").strip() or ".")
    if not os.path.exists(project):
        print(f"[AIDER] Path not found -> {project}")
        return
    print(f"[AIDER] Launching in {project} — type /exit to return\n")
    subprocess.run([AIDER_BIN], cwd=project)
    print("\n[AIDER] Back in local AI\n")

# ── Graph plotter ─────────────────────────────────────────
def try_plot(expression: str, title: str = ""):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        x_range = input("X range? (default: -10 10): ").strip() or "-10 10"
        parts = x_range.split()
        x = np.linspace(float(parts[0]), float(parts[1]), 1000)
        allowed = {
            "x": x, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "pi": np.pi,
            "e": np.e, "abs": np.abs, "sinh": np.sinh, "cosh": np.cosh,
            "tanh": np.tanh, "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan
        }
        y = eval(expression, {"__builtins__": {}}, allowed)
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, color="#00d4ff", linewidth=2)
        plt.axhline(0, color="#64748b", linewidth=0.8)
        plt.axvline(0, color="#64748b", linewidth=0.8)
        plt.grid(True, alpha=0.2, color="#1e2d3d")
        plt.title(title or expression, color="white", fontsize=12)
        plt.xlabel("x", color="white")
        plt.ylabel("y", color="white")
        plt.gca().set_facecolor("#0d1117")
        plt.gcf().set_facecolor("#070a0f")
        plt.tick_params(colors="white")
        for spine in plt.gca().spines.values():
            spine.set_edgecolor("#1e2d3d")
        plt.tight_layout()
        plt.savefig(GRAPH_FILE, dpi=150, facecolor="#070a0f")
        plt.show()
        print(f"[MATH] Graph saved -> {GRAPH_FILE}\n")
    except Exception as e:
        print(f"[MATH] Could not plot: {e}")

# ── Main loop ────────────────────────────────────────────
def main():
    memory = load_memory()
    profile = load_profile()

    print("🤖 Local AI ready. Type your question (or 'exit'):")
    if memory:
        print(f"   Memory: {len(memory)} exchanges | Profile: {'loaded' if profile else 'none'}\n")
    else:
        print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        # ── Built-in commands ──────────────────────────
        if user_input.lower() in ("help", "?", "commands"):
            print("""
┌─────────────────────────────────────────────────┐
│  LOCAL AI — COMMANDS & ROUTES                   │
├─────────────────────────────────────────────────┤
│  COMMANDS (type exactly)                        │
│  help / ?        show this list                 │
│  memory          show recent history            │
│  clear memory    wipe conversation history      │
│  profile         show your user profile         │
│  logs            list log files                 │
│  exit / quit     exit                           │
├─────────────────────────────────────────────────┤
│  ROUTES (just ask naturally)                    │
│  CODE     write / fix / explain code            │
│  MATH     equations, integrals, derivatives     │
│  SEARCH   current news, versions, prices        │
│  TTS      say "text" or speak "quoted text"     │
│  READ     read me the first page of X           │
│  FIXTEXT  fix my text "bad grammer here"        │
│  ANALYZE  analyze ~/path/to/file or folder      │
│  FILE     create a file / convert to pdf        │
│  AIDER    work on my project                    │
│  GENERAL  everything else                       │
└─────────────────────────────────────────────────┘
""")
            continue

        if user_input.lower() == "memory":
            print(f"\n[MEMORY] {len(memory)} exchanges in {MEMORY_FILE}")
            for m in memory[-5:]:
                print(f"  [{m.get('cat','?')}] Q: {m['q'][:60]}")
            print()
            continue

        if user_input.lower() == "logs":
            files = sorted(os.listdir(LOG_DIR)) if os.path.exists(LOG_DIR) else []
            print(f"\n[LOGS] {LOG_DIR}")
            for f in files:
                print(f"  {f}")
            print()
            continue

        if user_input.lower() == "profile":
            if os.path.exists(PROFILE_FILE):
                with open(PROFILE_FILE) as f:
                    print(f"\n[PROFILE]\n{json.dumps(json.load(f), indent=2)}\n")
            else:
                print("\n[PROFILE] No profile yet. Create ~/agents/profile.json\n")
            continue

        if user_input.lower() == "clear memory":
            memory = []
            save_memory(memory)
            print("[MEMORY] Cleared.\n")
            continue

        # ── Route ──────────────────────────────────────
        category = route(user_input)
        print(f"[Router -> {category}]\n")

        result = None

        if category == "CODE":
            ctx = memory_context(memory)
            desc = (f"{profile}\n\n{ctx}\n\nCoding request: {user_input}"
                    if (profile or ctx) else f"Coding request: {user_input}")
            agent = coder

        elif category == "MATH":
            agent = math_agent
            desc = (
                f"Solve step by step: {user_input}\n\n"
                "State the method, number each step, show all calculations, "
                "explain each step, state the final answer clearly. "
                "Mention any theorems or identities used."
            )

        elif category == "SEARCH":
            print("[Searching...]\n")
            search_results = searxng_search(user_input)
            desc = f"Search results:\n{search_results}\n\nAnswer this: {user_input}"
            agent = general

        elif category == "TTS":
            quoted = re.findall(r'"([^"]+)"', user_input)
            text = " ".join(quoted) if quoted else user_input
            for word in ["read this", "say", "speak", "read aloud", "tell me"]:
                text = text.replace(word, "")
            speak(text.strip())
            log(user_input, "[TTS]", category)
            continue

        elif category == "READ":
            read_agent(user_input)
            log(user_input, "[READ]", category)
            continue

        elif category == "ANALYZE":
            result = analyze_agent(user_input)
            if result:
                log(user_input, result, category)
                memory.append({"q": user_input, "a": result, "cat": category})
                save_memory(memory)
            continue

        elif category == "FIXTEXT":
            fix_input = user_input
            for phrase in ["fix my text", "correct this", "fix punctuation",
                           "fix grammar", "fix syntax", "fix"]:
                fix_input = fix_input.replace(phrase, "").strip()
            task = Task(
                description=f"Fix punctuation, grammar and syntax: '{fix_input}'",
                expected_output="Corrected text only.",
                agent=fixtext_agent
            )
            fixed = clean_output(str(Crew(agents=[fixtext_agent], tasks=[task], verbose=False).kickoff()).strip())
            print(f"\n[FIXTEXT] {fixed}\n")
            log(user_input, fixed, category)
            if input("Speak it? (y/n): ").strip().lower() == "y":
                speak(fixed)
            continue

        elif category == "FILE":
            file_agent(user_input)
            log(user_input, "[FILE]", category)
            continue

        elif category == "AIDER":
            run_aider(user_input)
            log(user_input, "[AIDER]", category)
            continue

        else:  # GENERAL
            ctx = memory_context(memory)
            parts = []
            if profile:
                parts.append(profile)
            if ctx:
                parts.append(ctx)
            parts.append(
                f"Answer this concisely. Use the profile and history above if relevant. "
                f"Never expose internal formatting: {user_input}"
            )
            desc = "\n\n".join(parts)
            agent = general

        task = Task(description=desc, expected_output="A helpful, concise response.", agent=agent)
        result = clean_output(str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip())
        print(f"\nAssistant: {result}\n")

        log(user_input, result, category)
        memory.append({"q": user_input, "a": result, "cat": category})
        save_memory(memory)

        if category == "MATH":
            plot = input("Plot a graph? (expression or blank to skip): ").strip()
            if plot:
                try_plot(plot, user_input)

if __name__ == "__main__":
    main()
