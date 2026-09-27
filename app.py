import os
import uvicorn

from typing import TypedDict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END


# ============================================================
# GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# HELPER
# ============================================================

def content_to_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                if "text" in item:
                    parts.append(str(item["text"]))

                elif "content" in item:
                    parts.append(str(item["content"]))

            elif hasattr(item, "text"):
                parts.append(str(item.text))

        return "\n".join(parts)

    if isinstance(content, dict):

        if "text" in content:
            return str(content["text"])

        if "content" in content:
            return str(content["content"])

    return str(content)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class DevCrewState(TypedDict, total=False):

    user_task: str

    plan: str

    code: str

    tests: str

    review: str

    final_solution: str


# ============================================================
# AGENT 1 - PLANNER
# ============================================================

def planner_agent(state: DevCrewState):

    prompt = ChatPromptTemplate.from_template("""
You are the PLANNER AGENT in an AI software development crew.

The user has given this development task:

{user_task}

Your job is to understand the requirement and create a
clear implementation plan.

Include:

1. Problem understanding
2. Required features
3. Technologies to use
4. File structure
5. Step-by-step implementation
6. Important edge cases

Do NOT write the complete code yet.

Keep the plan practical and clear.
""")

    chain = prompt | llm

    response = chain.invoke({
        "user_task": state["user_task"]
    })

    return {
        "plan": content_to_text(response.content)
    }


# ============================================================
# AGENT 2 - DEVELOPER
# ============================================================

def developer_agent(state: DevCrewState):

    prompt = ChatPromptTemplate.from_template("""
You are the DEVELOPER AGENT in an AI software development crew.

USER TASK:
{user_task}

PLANNER'S PLAN:
{plan}

Your job is to implement the solution.

Generate the actual code required for the task.

Requirements:

- Follow the planner's design.
- Write clean and readable code.
- Include necessary imports.
- Handle errors properly.
- Do not leave important parts as TODO.
- If multiple files are required, clearly separate them
  using filenames.

Return the implementation/code only with minimal explanation.
""")

    chain = prompt | llm

    response = chain.invoke({
        "user_task": state["user_task"],
        "plan": state["plan"]
    })

    return {
        "code": content_to_text(response.content)
    }


# ============================================================
# AGENT 3 - TESTER
# ============================================================

def tester_agent(state: DevCrewState):

    prompt = ChatPromptTemplate.from_template("""
You are the TESTER AGENT in an AI software development crew.

USER TASK:
{user_task}

IMPLEMENTATION:
{code}

Your job is to test and analyze the implementation.

Check for:

1. Syntax problems
2. Logical errors
3. Missing functionality
4. Edge cases
5. Security problems
6. Incorrect imports
7. Runtime issues
8. API or integration issues

Then provide:

- Problems found
- Why they are problems
- How to fix them
- Test cases that should be executed

If the implementation looks correct, clearly state that
and provide useful test cases.
""")

    chain = prompt | llm

    response = chain.invoke({
        "user_task": state["user_task"],
        "code": state["code"]
    })

    return {
        "tests": content_to_text(response.content)
    }


# ============================================================
# AGENT 4 - REVIEWER
# ============================================================

def reviewer_agent(state: DevCrewState):

    prompt = ChatPromptTemplate.from_template("""
You are the REVIEWER AGENT in an AI software development crew.

USER TASK:
{user_task}

PLANNER:
{plan}

DEVELOPER CODE:
{code}

TESTER REPORT:
{tests}

Review the complete solution.

Check:

1. Does the code satisfy the user's requirement?
2. Is the architecture reasonable?
3. Is the code maintainable?
4. Are there bugs?
5. Are there missing features?
6. Are the tester's concerns valid?
7. What should be improved?

Provide a final review with:

- Strengths
- Problems
- Required fixes
- Optional improvements
""")

    chain = prompt | llm

    response = chain.invoke({
        "user_task": state["user_task"],
        "plan": state["plan"],
        "code": state["code"],
        "tests": state["tests"]
    })

    return {
        "review": content_to_text(response.content)
    }


# ============================================================
# AGENT 5 - FINAL SOLUTION
# ============================================================

def final_solution_agent(state: DevCrewState):

    prompt = ChatPromptTemplate.from_template("""
You are the FINAL SOLUTION AGENT.

You are responsible for producing the final answer for the
software development task.

USER TASK:
{user_task}

PLANNER:
{plan}

DEVELOPER:
{code}

TESTER:
{tests}

REVIEWER:
{review}

Create the final solution.

Rules:

1. Apply the important fixes identified by the tester
   and reviewer.
2. Provide complete code.
3. Clearly show filenames when multiple files are needed.
4. Do not include incomplete TODO sections.
5. Keep the implementation practical.
6. Briefly explain how to run the solution.

Your response must be useful for a developer who wants to
actually implement the project.
""")

    chain = prompt | llm

    response = chain.invoke({
        "user_task": state["user_task"],
        "plan": state["plan"],
        "code": state["code"],
        "tests": state["tests"],
        "review": state["review"]
    })

    return {
        "final_solution": content_to_text(response.content)
    }


# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================

builder = StateGraph(DevCrewState)


builder.add_node(
    "planner",
    planner_agent
)

builder.add_node(
    "developer",
    developer_agent
)

builder.add_node(
    "tester",
    tester_agent
)

builder.add_node(
    "reviewer",
    reviewer_agent
)

builder.add_node(
    "final_solution",
    final_solution_agent
)


# ============================================================
# WORKFLOW CONNECTIONS
# ============================================================

builder.set_entry_point("planner")

builder.add_edge(
    "planner",
    "developer"
)

builder.add_edge(
    "developer",
    "tester"
)

builder.add_edge(
    "tester",
    "reviewer"
)

builder.add_edge(
    "reviewer",
    "final_solution"
)

builder.add_edge(
    "final_solution",
    END
)


graph = builder.compile()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="LangGraph AI Dev Crew",
    version="1.0"
)


# ============================================================
# API
# ============================================================

@app.post("/dev-crew")
async def run_dev_crew(data: dict):

    user_task = data.get("task", "").strip()

    if not user_task:

        return {
            "success": False,
            "error": "Please enter a development task."
        }

    initial_state = {
        "user_task": user_task
    }

    result = graph.invoke(initial_state)

    return {
        "success": True,
        "task": user_task,
        "plan": result.get("plan", ""),
        "code": result.get("code", ""),
        "tests": result.get("tests", ""),
        "review": result.get("review", ""),
        "final_solution": result.get(
            "final_solution",
            ""
        )
    }


# ============================================================
# FRONTEND
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>

<html>

<head>

<title>AI Dev Crew</title>

<meta name="viewport"
      content="width=device-width, initial-scale=1">

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f5f7fb;
    color: #111827;
}

.container {
    max-width: 1100px;
    margin: 40px auto;
    padding: 20px;
}

.header {
    text-align: center;
    margin-bottom: 30px;
}

.header h1 {
    font-size: 40px;
    margin-bottom: 10px;
}

.header p {
    color: #6b7280;
    font-size: 17px;
}

.card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
}

label {
    display: block;
    font-weight: bold;
    margin-bottom: 10px;
}

textarea {
    width: 100%;
    min-height: 150px;
    padding: 15px;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    font-size: 16px;
    resize: vertical;
}

button {
    width: 100%;
    margin-top: 20px;
    padding: 16px;
    border: none;
    border-radius: 12px;
    background: #4f46e5;
    color: white;
    font-size: 18px;
    font-weight: bold;
    cursor: pointer;
}

button:hover {
    background: #4338ca;
}

button:disabled {
    background: #9ca3af;
}

.loading {
    display: none;
    text-align: center;
    margin-top: 20px;
    color: #4f46e5;
    font-weight: bold;
}

.error {
    display: none;
    margin-top: 20px;
    padding: 15px;
    background: #fee2e2;
    color: #991b1b;
    border-radius: 10px;
}

.results {
    display: none;
    margin-top: 30px;
}

.agent-box {
    margin-top: 20px;
    padding: 20px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
}

.agent-box h3 {
    margin-top: 0;
}

.output {
    white-space: pre-wrap;
    line-height: 1.6;
    font-size: 15px;
}

.final {
    border: 2px solid #4f46e5;
    background: #eef2ff;
}

</style>

</head>


<body>

<div class="container">

<div class="header">

<h1>🤖 AI Dev Crew</h1>

<p>
Multi-Agent Software Development using LangGraph
</p>

</div>


<div class="card">

<form id="devForm">

<label>
💡 Describe your development task
</label>

<textarea
id="task"
placeholder="Example: Build a FastAPI REST API for a student management system with CRUD operations."
required
></textarea>


<button
id="runButton"
type="submit"
>
🚀 Start Dev Crew
</button>

</form>


<div
class="loading"
id="loading"
>
⏳ Dev Crew is working... Planner → Developer → Tester → Reviewer
</div>


<div
class="error"
id="error"
></div>


<div
class="results"
id="results"
>

<div class="agent-box">

<h3>🧠 Planner Agent</h3>

<div
class="output"
id="plan"
></div>

</div>


<div class="agent-box">

<h3>💻 Developer Agent</h3>

<div
class="output"
id="code"
></div>

</div>


<div class="agent-box">

<h3>🧪 Tester Agent</h3>

<div
class="output"
id="tests"
></div>

</div>


<div class="agent-box">

<h3>🔍 Reviewer Agent</h3>

<div
class="output"
id="review"
></div>

</div>


<div class="agent-box final">

<h3>🚀 Final Solution</h3>

<div
class="output"
id="finalSolution"
></div>

</div>

</div>

</div>

</div>


<script>

const form =
document.getElementById("devForm");

const button =
document.getElementById("runButton");

const loading =
document.getElementById("loading");

const errorBox =
document.getElementById("error");

const results =
document.getElementById("results");


form.addEventListener(
"submit",
async function(event) {

    event.preventDefault();

    const task =
    document.getElementById("task").value.trim();

    if (!task) {
        return;
    }

    button.disabled = true;

    button.innerText =
    "⏳ Dev Crew Running...";

    loading.style.display =
    "block";

    errorBox.style.display =
    "none";

    results.style.display =
    "none";


    try {

        const response =
        await fetch(
            "/dev-crew",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                    "application/json"
                },

                body: JSON.stringify({
                    task: task
                })
            }
        );


        const data =
        await response.json();


        if (!response.ok ||
            !data.success) {

            throw new Error(
                data.error ||
                "Something went wrong."
            );

        }


        document.getElementById(
            "plan"
        ).innerText =
        data.plan;


        document.getElementById(
            "code"
        ).innerText =
        data.code;


        document.getElementById(
            "tests"
        ).innerText =
        data.tests;


        document.getElementById(
            "review"
        ).innerText =
        data.review;


        document.getElementById(
            "finalSolution"
        ).innerText =
        data.final_solution;


        results.style.display =
        "block";


    } catch (error) {

        errorBox.innerText =
        "❌ " + error.message;

        errorBox.style.display =
        "block";

    }


    loading.style.display =
    "none";

    button.disabled =
    false;

    button.innerText =
    "🚀 Start Dev Crew";

});

</script>

</body>

</html>
"""


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
