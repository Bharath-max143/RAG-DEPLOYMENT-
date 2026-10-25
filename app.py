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
# Set GOOGLE_API_KEY in your .env file or deployment environment.
# Never hard-code API keys in source code.
API_KEY = os.getenv("AQ.Ab8RN6JrvCpgQt5md4rw9ZounGOuD9FVS9BbveST4eI472VBvA", "").strip().strip()
if not API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is missing. Add it to your .env file or deployment secrets."
    )

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash").strip()
TIMEOUT = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "45"))
MAX_ATTEMPTS = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))

import random
import time

def make_llm(model_name: str):
    """Create an LLM client; retries are managed in ask() to control fallback."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=API_KEY,
        temperature=0,
        max_output_tokens=2048,
        timeout=TIMEOUT,
        max_retries=0,
    )

# Clients are created lazily so an unavailable model does not prevent app startup.
_llm_clients = {}

def get_llm(model_name: str):
    if model_name not in _llm_clients:
        _llm_clients[model_name] = make_llm(model_name)
    return _llm_clients[model_name]

def is_transient_error(exc: Exception) -> bool:
    """Recognize temporary rate-limit/provider errors without depending on SDK internals."""
    message = str(exc).lower()
    markers = (
        "503", "unavailable", "429", "resource_exhausted",
        "500", "502", "504", "internal server error",
        "deadline exceeded", "temporarily", "timeout",
    )
    return any(marker in message for marker in markers)

def ask(prompt: str, **values) -> str:
    """Run an AI request with bounded retries and a configurable fallback model."""
    chain_prompt = ChatPromptTemplate.from_template(prompt)
    models_to_try = [MODEL]
    if FALLBACK_MODEL and FALLBACK_MODEL != MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_error = None
    for model_name in models_to_try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = (chain_prompt | get_llm(model_name)).invoke(values)
                content = result.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "\\n".join(
                        item.get("text", str(item)) if isinstance(item, dict) else str(item)
                        for item in content
                    )
                return str(content)
            except Exception as exc:
                last_error = exc
