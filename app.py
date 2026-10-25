import os
import re
import sys
import asyncio
from typing import TypedDict

# Ensure UTF-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END


# ---------- CONFIG ----------
API_KEY = os.getenv("GOOGLE_API_KEY", "AQ.Ab8RN6JrvCpgQt5md4rw9ZounGOuD9FVS9BbveST4eI472VBvA")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY before starting the app.")

# gemini-3.1-flash-lite is the active fast model
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
TIMEOUT = 60  # seconds per AI call

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=API_KEY,
    temperature=0,
    max_output_tokens=2048,
    timeout=TIMEOUT,
    max_retries=2,
)


# ---------- STATE ----------
class CrewState(TypedDict, total=False):
    user_task: str
    plan: str
    code: str
    tests: str
    review: str
    final_solution: str


def ask(prompt: str, **values) -> str:
    """Run one short AI request and return its text."""
    chain = ChatPromptTemplate.from_template(prompt) | llm
    result = chain.invoke(values)
    content = result.content

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


# ---------- AGENT 1: PLANNER ----------
def planner(state: CrewState):
    print("\n" + "=" * 70)
    print("🧠 [AGENT 1: PLANNER] Generating implementation plan...")
    print("=" * 70)
    result = ask(
        """You are a SOFTWARE PLANNER, not a career assistant.
Task: {task}

Create a concise implementation plan with:
features, technology, file structure, and key edge cases.
Maximum 180 words. Do not write the complete code.""",
        task=state["user_task"],
    )
    print(result.strip())
    return {"plan": result}


# ---------- AGENT 2: DEVELOPER ----------
def developer(state: CrewState):
    print("\n" + "=" * 70)
    print("💻 [AGENT 2: DEVELOPER] Generating code...")
    print("=" * 70)
    result = ask(
        """You are a SOFTWARE DEVELOPER.
Implement only the user's requested software task.

TASK:
{task}

PLAN:
{plan}

Return runnable code with filenames if multiple files are needed.
Include imports and essential error handling.
Avoid lengthy explanations and unrelated features.
Never answer with resume analysis or career advice.""",
        task=state["user_task"],
        plan=state["plan"],
    )
    print(result.strip())
    return {"code": result}


# ---------- AGENT 3: TESTER ----------
def tester(state: CrewState):
    print("\n" + "=" * 70)
    print("🧪 [AGENT 3: TESTER] Auditing code & generating tests...")
    print("=" * 70)
    result = ask(
        """You are a SOFTWARE TESTER.
Task: {task}

CODE:
{code}

Find only important syntax, logic, security, and requirement issues.
Give up to 5 concrete test cases and the necessary fixes.
Do NOT claim tests were executed; you have not run the code.
Maximum 250 words.""",
        task=state["user_task"],
        code=state["code"],
    )
    print(result.strip())
    return {"tests": result}


# ---------- AGENT 4: REVIEWER + FINAL SOLUTION ----------
def reviewer(state: CrewState):
    print("\n" + "=" * 70)
    print("🔍 [AGENT 4: REVIEWER & FINAL SOLUTION] Reviewing and finalizing...")
    print("=" * 70)
    result = ask(
        """You are the SOFTWARE REVIEWER and FINAL CODE EDITOR.

USER TASK:
{task}

PLAN:
{plan}

IMPLEMENTATION:
{code}

TESTER REPORT:
{tests}

Return these sections:
1. REVIEW: short list of important defects and fixes.
2. FINAL SOLUTION: corrected, complete code, with filenames if needed.
3. RUN: short setup and run commands.

Follow the user's task exactly. Do not generate career guidance,
resume analysis, job recommendations, or unrelated content.
Do not claim that code was executed unless it actually was.
Keep explanations concise but code complete.""",
        task=state["user_task"],
        plan=state["plan"],
        code=state["code"],
        tests=state["tests"],
    )
    print(result.strip())

    # Robust regex separation for FINAL SOLUTION header in various markdown formats
    match = re.search(r'(?:^|\n)(?:[#*`\d.\s-]*)FINAL\s+SOLUTION\s*:?[#*`\s]*\n', result, re.IGNORECASE)
    if match:
        review_text = result[:match.start()].strip()
        final_text = result[match.end():].strip()
        return {
            "review": review_text,
            "final_solution": final_text,
        }

    # Inline fallback
    marker_match = re.search(r'FINAL\s+SOLUTION\s*:', result, re.IGNORECASE)
    if marker_match:
        review_text = result[:marker_match.start()].strip()
        final_text = result[marker_match.end():].strip()
        return {
            "review": review_text,
            "final_solution": final_text,
        }

    # Safe fallback if the model does not include the exact header
    return {
        "review": result,
        "final_solution": state.get("code", ""),
    }


# ---------- LANGGRAPH ----------
workflow = StateGraph(CrewState)

workflow.add_node("planner", planner)
workflow.add_node("developer", developer)
workflow.add_node("tester", tester)
workflow.add_node("reviewer", reviewer)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "reviewer")
workflow.add_edge("reviewer", END)

graph = workflow.compile()


# ---------- FASTAPI ----------
app = FastAPI(title="AI Dev Crew", version="2.0")


class TaskRequest(BaseModel):
    task: str = Field(min_length=3, max_length=4000)


@app.get("/health")
def health():
    return {"status": "ok", "app": "AI Dev Crew", "version": "2.0", "model": MODEL}


@app.post("/dev-crew")
async def run_dev_crew(request: TaskRequest):
    task = request.task.strip()

    if len(task) < 3:
        raise HTTPException(
            status_code=400,
            detail="Please enter a development task with at least 3 characters.",
        )

    try:
        # Run synchronous LangGraph outside the async event loop
        result = await asyncio.wait_for(
            asyncio.to_thread(
                graph.invoke,
                {"user_task": task},
            ),
            timeout=240,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="The AI workflow timed out. Try a simpler task or check your connection.",
        )
    except Exception as exc:
        print(f"Dev Crew error: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"AI workflow failed: {str(exc)}",
        )

    return {
        "success": True,
        "task": task,
        "plan": result.get("plan", ""),
        "code": result.get("code", ""),
        "tests": result.get("tests", ""),
        "review": result.get("review", ""),
        "final_solution": result.get("final_solution", ""),
    }


# ---------- SIMPLE FRONTEND ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Dev Crew</title>
<style>
* { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    max-width: 1000px;
    margin: 40px auto;
    padding: 20px;
}
.header { text-align: center; margin-bottom: 30px; }
.header h1 { font-size: 2.2rem; margin-bottom: 8px; color: #f8fafc; }
.header p { color: #94a3b8; font-size: 1.05rem; }
.badge { display: inline-block; padding: 4px 12px; background: #1e293b; border: 1px solid #334155; border-radius: 20px; font-size: 0.85rem; color: #38bdf8; margin-top: 6px; }
.card {
    background: #1e293b;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #334155;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
label { font-weight: 600; font-size: 1.05rem; color: #f1f5f9; display: block; margin-bottom: 10px; }
textarea {
    width: 100%;
    height: 120px;
    padding: 14px;
    font-size: 15px;
    background: #0f172a;
    color: #f8fafc;
    border: 1px solid #475569;
    border-radius: 10px;
    resize: vertical;
    outline: none;
    transition: border-color 0.2s;
}
textarea:focus { border-color: #6366f1; }
button {
    width: 100%;
    padding: 15px;
    margin-top: 15px;
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    border: 0;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.1s, opacity 0.2s;
}
button:hover:not(:disabled) { transform: translateY(-1px); }
button:disabled { opacity: 0.6; cursor: not-allowed; }
#status { margin-top: 15px; font-weight: 500; min-height: 24px; text-align: center; }
.agent {
    margin-top: 20px;
    padding: 20px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
}
.agent h3 {
    margin-top: 0;
    margin-bottom: 12px;
    font-size: 1.15rem;
    color: #38bdf8;
    display: flex;
    align-items: center;
    gap: 8px;
}
.output {
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.6;
    background: #0f172a;
    padding: 16px;
    border-radius: 8px;
    border: 1px solid #334155;
    font-family: Consolas, Monaco, "Courier New", monospace;
    font-size: 14px;
    color: #f1f5f9;
    max-height: 500px;
    overflow-y: auto;
}
</style>
</head>
<body>
<div class="header">
    <h1>🤖 AI Dev Crew</h1>
    <p>Multi-Agent Software Development Squad with LangGraph & Google Gemini</p>
    <div class="badge">Model: gemini-3.1-flash-lite</div>
</div>

<div class="card">
    <label for="task">Describe your development task</label>
    <textarea id="task" placeholder="e.g. Build a Python CLI password manager using SQLite and cryptography">Build a FastAPI student CRUD API using SQLite</textarea>
    <button id="run" onclick="runCrew()">🚀 Start Dev Crew</button>
    <div id="status"></div>
</div>

<div id="results" hidden>
    <div class="agent"><h3>🧠 Agent 1: Planner</h3><div class="output" id="plan"></div></div>
    <div class="agent"><h3>💻 Agent 2: Developer</h3><div class="output" id="code"></div></div>
    <div class="agent"><h3>🧪 Agent 3: Tester</h3><div class="output" id="tests"></div></div>
    <div class="agent"><h3>🔍 Agent 4: Reviewer</h3><div class="output" id="review"></div></div>
    <div class="agent"><h3>🚀 Final Solution</h3><div class="output" id="final_solution"></div></div>
</div>

<script>
async function runCrew() {
    const task = document.getElementById("task").value.trim();
    const btn = document.getElementById("run");
    const status = document.getElementById("status");
    const results = document.getElementById("results");

    if (task.length < 3) {
        status.innerHTML = "<span style='color:#f87171'>⚠️ Enter a task with at least 3 characters.</span>";
        return;
    }

    btn.disabled = true;
    status.innerHTML = "<span style='color:#38bdf8'>⏳ Agents are planning, writing code, testing, and reviewing... Please wait.</span>";
    results.hidden = true;

    try {
        const response = await fetch("/dev-crew", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Request failed");

        for (const key of ["plan", "code", "tests", "review", "final_solution"]) {
            document.getElementById(key).textContent = data[key] || "No output returned.";
        }
        results.hidden = false;
        status.innerHTML = "<span style='color:#4ade80'>✅ Dev Crew workflow completed successfully!</span>";
    } catch (e) {
        status.innerHTML = "<span style='color:#f87171'>❌ " + e.message + "</span>";
    } finally {
        btn.disabled = false;
    }
}
</script>
</body>
</html>
"""


# ---------- CLI EXECUTION ----------
def run_cli(task: str):
    print("\n" + "=" * 70)
    print("🚀 AI DEV CREW - RUNNING LIVE WORKFLOW")
    print(f"📌 Task: {task}")
    print(f"🤖 Model: {MODEL}")
    print("=" * 70)

    try:
        final_state = graph.invoke({"user_task": task})
        print("\n" + "=" * 70)
        print("✅ WORKFLOW COMPLETE!")
        print("=" * 70)
        return final_state
    except Exception as e:
        print(f"\n❌ Execution error: {e}")
        return None


if __name__ == "__main__":
    server_flags = {"--server", "--serve", "-s", "server", "serve"}

    # If flags match server flags, or if no arguments are passed, start web server
    if len(sys.argv) == 1 or any(arg in server_flags for arg in sys.argv[1:]):
        port = int(os.getenv("PORT", "8000"))
        print(f"🚀 AI Dev Crew web server starting on http://localhost:{port}")
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--cli":
        task_input = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Build a Python utility to validate and sanitize email addresses"
        run_cli(task_input)
    else:
        # User passed a custom task string directly, e.g. python main.py "task description"
        task_input = " ".join(sys.argv[1:])
        run_cli(task_input)
