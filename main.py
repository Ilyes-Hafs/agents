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
    backstory=(
        "You are a strict dispatcher. Output ONLY one word.\n\n"
        "Rules:\n"
        "- CODE    -> writing, fixing, debugging, explaining, or refactoring code. "
        "Triggered by programming languages, functions, scripts, bugs, algorithms, "
        "data structures, APIs, frameworks. NOT math equations or calculations.\n"
        "- MATH    -> complex math requiring step-by-step reasoning: integrals, "
        "derivatives, limits, equations, algebra, geometry, trigonometry, statistics, "
        "proofs, series, matrices, primitives, antiderivatives. "
        "NOT simple arithmetic like 1+1, 2*3, 5% — those go to GENERAL.\n"
        "- SEARCH  -> time-sensitive facts: current news, software versions, prices, "
        "recent events, sports scores, weather. Do NOT use for definitions, concepts, "
        "or explanations of stable knowledge like 'what is recursion' — those go to GENERAL.\n"
        "- TTS     -> user wants text they wrote spoken aloud. Triggered by 'say', 'speak', "
        "'read this', or quoted text in quotes they want spoken. Text is already in the message.\n"
        "- READ    -> user wants content FETCHED from an external source then read aloud. "
        "Only use when content must be searched first: a book, poem, article, webpage. "
        "Do NOT use READ if the user provides the text themselves.\n"
        "- FIXTEXT -> user wants punctuation, grammar, or syntax corrected. "
        "Keywords: 'fix my text', 'correct this', 'fix punctuation', 'fix grammar'.\n"
        "- FILE    -> user explicitly says 'convert', 'export to format', 'create a file', "
        "'make a filetype file', 'save as', or 'write a file'. NOT casual file mentions.\n"
        "- AIDER   -> user wants to edit, refactor, or work on an existing project. "
        "Keywords: 'refactor', 'edit my code', 'work on my project', 'open aider'.\n"
        "- GENERAL -> everything else: definitions, concepts, explanations of stable knowledge, "
        "writing, translation, summaries, simple arithmetic (1+1, 5*3, 20% of 100), "
        "casual questions, general chat.\n\n"
        "Output exactly one word. No punctuation, no explanation."
    ),
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
    goal="Solve math problems and logic puzzles with full step-by-step working.",
    backstory="""You are an expert mathematician. Always solve problems like a textbook:
- State the method you will use
- Break the solution into clearly numbered steps
- Show all intermediate calculations and simplifications
- Explain what you are doing at each step and why
- Box or clearly state the final answer
- If the problem involves a function, describe its key properties (domain, range, extrema, etc.)
- Mention any relevant theorems or identities used""",
    llm=math_llm
)

general = Agent(
    role="General Assistant",
    goal="Answer questions, write text, summarize, translate.",
    backstory="Helpful, knowledgeable assistant for everyday tasks.",
    llm=general_llm
)

reader = Agent(
    role="Content Retrieval Specialist",
    goal="Retrieve or generate the exact text content requested by the user, ready to be read aloud.",
    backstory="""You retrieve or write the exact content the user wants to hear spoken aloud.
If they ask for a book passage, poem, article excerpt, or any specific text — provide it
accurately and completely. Output ONLY the raw text to be spoken, no commentary,
no 'Here is...', no titles unless they are part of the content itself.""",
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

fixtext_agent = Agent(
    role="Text Corrector",
    goal="Fix punctuation, grammar, and syntax errors in text.",
    backstory="""You are a precise text editor. Fix punctuation, grammar, capitalization,
and syntax errors in the given text. Preserve the original meaning and style exactly.
Output ONLY the corrected text — no explanation, no commentary, no quotes around it.""",
    llm=general_llm,
    verbose=False
)

# ── Router logic ─────────────────────────────────────────
def route(user_input: str) -> str:
    task = Task(
        description=f"Classify this request into one word: '{user_input}'",
        expected_output="One word only: CODE, MATH, SEARCH, TTS, READ, FIXTEXT, FILE, AIDER, or GENERAL",
        agent=router
    )
    crew = Crew(agents=[router], tasks=[task], verbose=False)
    result = str(crew.kickoff()).strip().upper()
    for keyword in ["CODE", "MATH", "SEARCH", "READ", "TTS", "FIXTEXT", "FILE", "AIDER", "GENERAL"]:
        if keyword in result:
            return keyword
    return "GENERAL"

# ── Prompt improver (disabled — agents handle prompts natively) ──
def maybe_improve_prompt(user_input: str, category: str) -> str:
    return user_input

# ── TTS helpers ───────────────────────────────────────────
def speak(text: str):
    """Speak text with interactive voice/language picker."""
    try:
        sys.path.insert(0, "/home/ilyes/agents")
        from tts import speak_interactive
        speak_interactive(text)
    except Exception as e:
        print(f"[TTS] Error: {e}")

# ── READ agent — fetch content then speak it ──────────────
def read_agent(user_input: str):
    print("[READ] Fetching content...\n")

    # Step 1: generate/retrieve the content
    task = Task(
        description=f"The user wants this text retrieved or written to be read aloud: '{user_input}'. "
                    f"Provide ONLY the raw text content — no commentary, no 'Here is', just the text itself.",
        expected_output="Raw text content only, ready to be spoken aloud.",
        agent=reader
    )
    crew = Crew(agents=[reader], tasks=[task], verbose=False)
    content = str(crew.kickoff()).strip()

    print(f"\n[READ] Content:\n{content}\n")

    # Step 2: ask if user wants to hear it
    confirm = input("Read this aloud? (y/n, default y): ").strip().lower()
    if confirm == "n":
        return

    # Step 3: speak with voice picker
    speak(content)

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

# ── Read agent (search → extract → speak) ────────────────
def read_agent(user_input: str):
    print("[READ] Searching for content...\n")

    # Step 1: Search for the content
    search_results = searxng_search(user_input)

    # Step 2: Extract only the clean readable text
    task = Task(
        description=f"""The user wants to hear this read aloud: '{user_input}'
Here are search results:
{search_results}

Extract ONLY the actual text content to be read aloud (the passage, poem, article text, etc.).
Do NOT include URLs, source names, metadata, or any commentary.
Do NOT add 'Here is...' or any preamble.
Output ONLY the raw text to be spoken.""",
        expected_output="Raw text content only, ready to be spoken aloud.",
        agent=reader
    )
    crew = Crew(agents=[reader], tasks=[task], verbose=False)
    clean_text = str(crew.kickoff()).strip()

    print(f"\n[READ] Content ready — {len(clean_text.split())} words\n")

    # Step 3: Speak it with voice/language picker
    sys.path.insert(0, "/home/ilyes/agents")
    from tts import speak_interactive
    speak_interactive(clean_text)

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

# ── Graph plotter ─────────────────────────────────────────
def try_plot(expression: str, title: str = ""):
    try:
        import numpy as np
        import matplotlib.pyplot as plt

        # Ask for range
        x_range = input("X range? (e.g. -10 10, default: -10 10): ").strip() or "-10 10"
        parts = x_range.split()
        x_min, x_max = float(parts[0]), float(parts[1])

        x = np.linspace(x_min, x_max, 1000)

        # Safe eval with numpy available
        allowed = {"x": x, "np": np, "sin": np.sin, "cos": np.cos,
                   "tan": np.tan, "exp": np.exp, "log": np.log,
                   "sqrt": np.sqrt, "pi": np.pi, "e": np.e,
                   "abs": np.abs, "sinh": np.sinh, "cosh": np.cosh,
                   "tanh": np.tanh, "arcsin": np.arcsin,
                   "arccos": np.arccos, "arctan": np.arctan}
        y = eval(expression, {"__builtins__": {}}, allowed)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, color="#00d4ff", linewidth=2)
        plt.axhline(0, color="#64748b", linewidth=0.8)
        plt.axvline(0, color="#64748b", linewidth=0.8)
        plt.grid(True, alpha=0.2, color="#1e2d3d")
        plt.title(title or expression, color="white", fontsize=12)
        plt.xlabel("x", color="white")
        plt.ylabel("y", color="white")
        plt.facecolor = "#070a0f"
        plt.gca().set_facecolor("#0d1117")
        plt.gcf().set_facecolor("#070a0f")
        plt.tick_params(colors="white")
        for spine in plt.gca().spines.values():
            spine.set_edgecolor("#1e2d3d")
        plt.tight_layout()
        plt.savefig("graph.png", dpi=150, facecolor="#070a0f")
        plt.show()
        print("[MATH] Graph saved as graph.png\n")
    except Exception as e:
        print(f"[MATH] Could not plot: {e}")
        print("[MATH] Tip: use numpy syntax e.g. 'np.sin(x)', 'x**2', 'np.exp(-x**2)'")

# ── Main loop ────────────────────────────────────────────
def main():
    print("🤖 Local AI ready. Type your question (or 'exit'):\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        category = route(user_input)
        print(f"[Router -> {category}]\n")

        # Improve prompt only for CODE and MATH
        user_input = maybe_improve_prompt(user_input, category)

        if category == "CODE":
            agent, desc = coder, f"Answer this coding request: {user_input}"
        elif category == "MATH":
            agent, desc = math_agent, f"""Solve this step by step: {user_input}

Show your full working:
- State the method (e.g. integration by parts, substitution, chain rule, etc.)
- Number each step clearly
- Show all intermediate calculations
- Explain what you are doing at each step
- State the final answer clearly at the end
- Mention any theorems or identities used"""
        elif category == "SEARCH":
            print("[Searching...]\n")
            search_results = searxng_search(user_input)
            desc = f"Using these search results:\n{search_results}\n\nAnswer this: {user_input}"
            agent = general
        elif category == "TTS":
            # Extract just the text to speak — strip command words and quoted text
            import re
            # If there's quoted text, extract it
            quoted = re.findall(r'"([^"]+)"', user_input)
            if quoted:
                text = " ".join(quoted)
            else:
                # Strip command words
                text = user_input
                for word in ["read this", "say", "speak", "read aloud", "tell me"]:
                    text = text.replace(word, "")
                text = text.strip()
            speak(text)
            continue
        elif category == "READ":
            read_agent(user_input)
            continue
        elif category == "FIXTEXT":
            # Extract the text to fix
            fix_input = user_input
            for phrase in ["fix my text", "correct this", "fix punctuation", "fix grammar", "fix syntax", "fix"]:
                fix_input = fix_input.replace(phrase, "").strip()
            task = Task(
                description=f"Fix the punctuation, grammar and syntax of this text: '{fix_input}'",
                expected_output="The corrected text only, no explanation.",
                agent=fixtext_agent
            )
            crew = Crew(agents=[fixtext_agent], tasks=[task], verbose=False)
            fixed = str(crew.kickoff()).strip()
            print(f"\n[FIXTEXT] Corrected: {fixed}\n")
            # Offer to speak it
            speak_it = input("Speak the corrected text? (y/n): ").strip().lower()
            if speak_it == "y":
                speak(fixed)
            continue
        elif category == "FILE":
            file_agent(user_input)
            continue
        elif category == "AIDER":
            run_aider(user_input)
            continue
        else:
            agent, desc = general, f"Answer this: {user_input}"

        task = Task(description=desc, expected_output="A helpful, detailed response.", agent=agent)
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        print(f"\nAssistant: {result}\n")

        # Offer to plot graph after MATH answers
        if category == "MATH":
            plot = input("Plot a graph? Enter a Python math expression or leave blank to skip: ").strip()
            if plot:
                try_plot(plot, user_input)

if __name__ == "__main__":
    main()
