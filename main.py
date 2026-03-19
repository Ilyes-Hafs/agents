from crewai import Agent, Task, Crew, LLM
import subprocess
import os
import sys
import requests

# ── Models ──────────────────────────────────────────────
router_llm  = LLM(model="ollama/qwen2.5:3b",          base_url="http://localhost:11434")
coder_llm   = LLM(model="ollama/qwen2.5-coder:7b",    base_url="http://localhost:11434")
math_llm    = LLM(model="ollama/deepseek-r1:7b",       base_url="http://localhost:11434")
general_llm = LLM(model="ollama/llama3.2:3b",          base_url="http://localhost:11434")

# ── SearXNG search function ──────────────────────────────
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

# ── Agents ───────────────────────────────────────────────
router = Agent(
    role="Router",
    goal="Classify the user request into one category.",
    backstory="""You are a strict dispatcher. Output ONLY one word.

Rules:
- CODE    -> writing, fixing, explaining, or debugging code
- MATH    -> calculations, equations, logic problems
- SEARCH  -> current events, facts that need a web lookup
- TTS     -> user says 'say', 'speak', 'read aloud', 'read this'
- FILE    -> user explicitly says 'convert', 'export to [format]',
            'create a file', 'make a [filetype] file', 'save as',
            or 'write a file'. NOT triggered by casual file mentions.
- AIDER   -> user wants to edit, refactor, fix, or work on an existing
            project or codebase. Keywords: 'refactor', 'edit my code',
            'work on my project', 'open aider', 'fix my project'.
- GENERAL -> everything else

Output exactly one word. No punctuation, no explanation.""",
    llm=router_llm,
    verbose=False
)

coder = Agent(
    role="Coding Specialist",
    goal="Write, explain, debug, and improve code.",
    backstory="Expert software engineer. You write clean, working code and explain it clearly.",
    llm=coder_llm
)

math_agent = Agent(
    role="Math & Reasoning Specialist",
    goal="Solve math problems and logic puzzles step by step.",
    backstory="Expert mathematician and logical thinker.",
    llm=math_llm
)

general = Agent(
    role="General Assistant",
    goal="Answer questions, write text, summarize, translate.",
    backstory="Helpful, knowledgeable assistant for everyday tasks.",
    llm=general_llm
)

prompter = Agent(
    role="Prompt Engineer",
    goal="Rewrite vague or short user inputs into clear, detailed, well-structured prompts.",
    backstory="""You are an expert prompt engineer. When given a vague or short request,
rewrite it as a clear, specific, and well-structured prompt that will get the best possible
result from an AI model. Preserve the original intent exactly. Output ONLY the improved
prompt — no explanation, no preamble, no quotes around it.""",
    llm=general_llm,
    verbose=False
)

# ── Router logic ─────────────────────────────────────────
def route(user_input: str) -> str:
    task = Task(
        description=f"Classify this request into one word: '{user_input}'",
        expected_output="One word only: CODE, MATH, SEARCH, TTS, FILE, AIDER, or GENERAL",
        agent=router
    )
    crew = Crew(agents=[router], tasks=[task], verbose=False)
    result = str(crew.kickoff()).strip().upper()
    for keyword in ["CODE", "MATH", "SEARCH", "TTS", "FILE", "AIDER", "GENERAL"]:
        if keyword in result:
            return keyword
    return "GENERAL"

# ── Prompt improver ───────────────────────────────────────
def maybe_improve_prompt(user_input: str) -> str:
    # Only improve if message is short or vague (under 10 words)
    if len(user_input.split()) < 10:
        task = Task(
            description=f"Rewrite this as a clearer, more detailed prompt: '{user_input}'",
            expected_output="An improved prompt only, no explanation.",
            agent=prompter
        )
        crew = Crew(agents=[prompter], tasks=[task], verbose=False)
        improved = str(crew.kickoff()).strip()
        if improved and improved != user_input:
            print(f"[PROMPT] -> {improved}\n")
            return improved
    return user_input

# ── TTS ──────────────────────────────────────────────────
def speak(text: str):
    try:
        import sounddevice as sd
        from kokoro_onnx import Kokoro
        kokoro = Kokoro("/home/ilyes/agents/kokoro-v0_19.onnx", "/home/ilyes/agents/voices-v1.0.bin")
        samples, sample_rate = kokoro.create(text, voice="am_fenrir", speed=1.2, lang="en-us")
        sd.play(samples, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"[TTS] Error: {e}")

# ── File agent (create + convert) ────────────────────────
CONVERT_KEYWORDS = ["convert", "export", "change format", "turn into", "transform", "save as"]

def file_agent(user_input: str):
    is_convert = any(kw in user_input.lower() for kw in CONVERT_KEYWORDS)

    if is_convert:
        print("[FILE] Convert mode")
        src = input("Source file path: ").strip()
        src = os.path.expanduser(src)

        if not os.path.exists(src):
            print(f"[FILE] Error: file not found -> {src}")
            return

        fmt = input("Target format (pdf/docx/txt/html/md): ").strip().lower()
        out = os.path.splitext(src)[0] + "." + fmt

        cmd = ["pandoc", src, "-o", out]
        if fmt == "pdf":
            cmd += ["--pdf-engine=xelatex"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[FILE] Done -> {out}")
        else:
            print(f"[FILE] Error:\n{result.stderr}")

    else:
        print("[FILE] Create mode")
        filename = input("Filename (e.g. notes.md, script.py, hello.txt): ").strip()
        if not filename:
            print("[FILE] No filename given, cancelled.")
            return

        save_path = input("Save to folder (default: ~/Documents): ").strip() or "~/Documents"
        save_path = os.path.expanduser(save_path)
        os.makedirs(save_path, exist_ok=True)

        ext = os.path.splitext(filename)[1].lower()

        generate = input("Generate content with AI? (y/n): ").strip().lower()
        if generate == "y":
            prompt = input("Describe what the file should contain: ").strip() or user_input
            agent = coder if ext in [".py", ".js", ".ts", ".sh", ".c", ".cpp", ".rs", ".go"] else general
            task = Task(
                description=f"Write the full content for a file called '{filename}'. Request: {prompt}. Output ONLY the file content, no explanation.",
                expected_output="Raw file content only.",
                agent=agent
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            content = str(crew.kickoff()).strip()
        else:
            print("Enter content (type END on a new line when done):")
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

        fmt = input("Also convert to another format? (pdf/docx/leave blank to skip): ").strip().lower()
        if fmt:
            out = os.path.splitext(full_path)[0] + "." + fmt
            cmd = ["pandoc", full_path, "-o", out]
            if fmt == "pdf":
                cmd += ["--pdf-engine=xelatex"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[FILE] Also converted -> {out}")
            else:
                print(f"[FILE] Convert error:\n{result.stderr}")

# ── Aider agent ──────────────────────────────────────────
AIDER_BIN = "/home/ilyes/.venvs/aider/bin/aider"

def run_aider(user_input: str):
    project = input("Project path (default: current dir): ").strip() or "."
    project = os.path.expanduser(project)
    if not os.path.exists(project):
        print(f"[AIDER] Path not found -> {project}")
        return
    print(f"[AIDER] Launching in {project} — type /exit to return to agents\n")
    subprocess.run([AIDER_BIN], cwd=project)
    print("\n[AIDER] Back in local AI\n")

# ── Main loop ────────────────────────────────────────────
def main():
    print("🤖 Local AI ready. Type your question (or 'exit'):\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        # Auto-improve short/vague inputs before routing
        user_input = maybe_improve_prompt(user_input)

        category = route(user_input)
        print(f"[Router -> {category}]\n")

        if category == "CODE":
            agent, desc = coder, f"Answer this coding request: {user_input}"
        elif category == "MATH":
            agent, desc = math_agent, f"Solve this: {user_input}"
        elif category == "SEARCH":
            print("[Searching...]\n")
            search_results = searxng_search(user_input)
            desc = f"Using these search results:\n{search_results}\n\nAnswer this: {user_input}"
            agent = general
        elif category == "TTS":
            speak(user_input.replace("read this", "").replace("say", "").strip())
            continue
        elif category == "FILE":
            file_agent(user_input)
            continue
        elif category == "AIDER":
            run_aider(user_input)
            continue
        else:
            agent, desc = general, f"Answer this: {user_input}"

        task = Task(description=desc, expected_output="A helpful response.", agent=agent)
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        print(f"\nAssistant: {result}\n")

if __name__ == "__main__":
    main()
