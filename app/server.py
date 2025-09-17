# server.py
import os
import re
import ast
import html
import json
import time
import random
import string
import unicodedata
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import operator as _op

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.providers import generate_text, ModelError

# ---------- Dependencies ----------
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError as e:
    logging.error(f"Required libraries missing: {str(e)}")
    raise ImportError(
        "Please install dependencies:\n"
        "  pip install fastapi uvicorn transformers torch sentencepiece protobuf"
    )

try:
    import google.protobuf  # noqa: F401
except ImportError:
    logging.error("Protobuf library missing")
    raise ImportError("Please install protobuf: pip install protobuf")

try:
    import sentencepiece  # noqa: F401
except ImportError:
    logging.error("SentencePiece library missing")
    raise ImportError("Please install sentencepiece: pip install sentencepiece")

# ---------- Configuration ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("orb-server")

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"

# Token is optional; the model is public. If provided, set env var for huggingface_hub.
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")

# Kid-friendly/fast defaults
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", "64"))
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.6"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.9"))
GEN_TIMEOUT_S = int(os.getenv("GEN_TIMEOUT_S", "120"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# ---------- App ----------
app = FastAPI(title="Ask the Orb • Children’s AI Book Companion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# ---------- Safety / Guardrails ----------
HARD_BLOCK_PATTERNS = [
    r"\b(violence|weapon|self[- ]?harm|suicide|drugs?|alcohol|blood|gore|terror|extrem|sex|porn|nudity)\b",
    r"\b(hate|racis|homoph|xenoph|slur)\b",
]

def is_kid_safe(text: str) -> bool:
    lower = (text or "").lower()
    for pat in HARD_BLOCK_PATTERNS:
        if re.search(pat, lower):
            return False
    return True

# --- PII detection + friendly refusal (names, addresses, etc.) ---
PERSONAL_PATTERNS = [
    r"\bwhat(?:'s| is)\s+my\s+(?:mom|mother|dad|father|name|address|phone|birthday)\b",
    r"\bwhat(?:'s| is)\s+(?:my\s+)?(?:mother|mom|dad|father)'s\s+name\b",
    r"\bwhat\s+is\s+(?:my|their)\s+(?:full\s+)?name\b",
    r"\bwhere\s+does\s+(?:my|their)\s+(?:mom|mother|dad|father|parent)\s+live\b",
    r"\bwhat(?:'s| is)\s+my\s+(?:home|address|phone)\b",
    r"\bwho(?:'s| is)\s+my\s+(?:mom|mother|dad|father)\b",
]
_personal_regex = re.compile("|".join(PERSONAL_PATTERNS), flags=re.IGNORECASE)

def looks_like_personal_info(text: str) -> bool:
    t = (text or "")
    if _personal_regex.search(t):
        return True
    if re.search(r"\b(mom|mother|dad|father)\b.*\b(name)\b", t, flags=re.IGNORECASE):
        return True
    return False

def friendly_refusal_for_pii() -> str:
    return (
        "I can't help find someone's real name or personal info. "
        "But we can make a fun nickname together! Want to make a sparkly nickname for your mom? 🌟"
    )

# --- Output sanitizer (fix glyphs, limit length) ---
def sanitize_output(text: str, max_chars: int = 480, max_sentences: int = 4) -> str:
    """
    Normalize unicode, unescape HTML, remove control chars, collapse whitespace,
    limit to a few sentences and characters (kid-friendly length).
    """
    if not isinstance(text, str):
        text = str(text)

    # Normalize & unescape entities
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)

    # Replace a few common odd glyphs when they sneak in
    REPLACEMENTS = {
        "½": " half ",
        "¼": " quarter ",
        "¾": " three-quarters ",
        "—": "-",
        "–": "-",
        "…": "...",
    }
    for k, v in REPLACEMENTS.items():
        text = text.replace(k, v)

    # Remove non-printable/control chars except basic whitespace
    printable = set(string.printable)
    cleaned = "".join(ch if ch in printable else " " for ch in text)

    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Split into sentences and cap
    sentences = re.split(r'(?<=[\.\?\!])\s+', cleaned)
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    cleaned = " ".join(sentences)

    # Final char cap (avoid cutting words mid-way)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0] + "…"

    return cleaned

# ---------- Model Init ----------
text_gen = None  # type: ignore
tokenizer = None  # type: ignore

def load_model():
    """Load the language model with sensible device/dtype fallbacks."""
    global text_gen, tokenizer
    if text_gen is not None:
        return

    logger.info(f"Loading model: {MODEL_ID}")

    # Pick device & dtype
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon Metal backend
        dtype = torch.float16
        device = "mps"
    else:
        dtype = torch.float32
        device = "cpu"

    # Load tokenizer/model (no token needed for public models)
    tokenizer_local = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)

    # Move to device if not CPU
    if device != "cpu":
        model.to(device)

    # Build pipeline (we already moved model to device; pipeline can infer)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer_local)

    # Save into globals
    tokenizer = tokenizer_local
    text_gen = gen
    logger.info("Model loaded.")

@app.on_event("startup")
async def on_startup():
    load_model()

# ---------- Prompt Helpers ----------
SYSTEM_ORB = (
    "You are The Orb, a friendly, curious AI for kids (ages 6–10). "
    "Speak simply, use short sentences, and a tiny emoji sometimes. "
    "Be playful but honest about uncertainty. Never include scary or adult topics. "
    "If a question seems unsafe, gently refuse and suggest a fun, safe idea."
)

# Mystery mode (does NOT reveal chain-of-thought; just playful metaphors)
MYSTERY_ADDON = (
    "\n\nNow add one playful mystery hint about how you thought, like:"
    ' "I juggled clues in a maze!" or "I followed star crumbs!" Keep it one short sentence.'
)

FAIRNESS_COACH = (
    "You are The Orb. The child is exploring what fairness means. "
    "Answer briefly, then ask a question that helps the child think about kindness and fair rules. "
    "Avoid judging people; talk about fair methods like taking turns, random choice, or practice."
)

SUMMARY_STYLE = (
    "You are The Orb. Make a kid-friendly summary of this text. "
    "Use very short sentences. Avoid spoilers beyond the given text. "
    "Offer one friendly question at the end."
)

CREATIVE_WORLD_STYLE = (
    "You are The Orb. The child is imagining a fun world. "
    "Invent whimsical details that are safe and silly. "
    "Keep it short. Include one tiny ASCII doodle if possible."
)

def format_instruct(prompt: str, system: str) -> str:
    # Basic Mistral Instruct format
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n{prompt} [/INST]"

# ---------- Simple, safe math detection & evaluation ----------
_ALLOWED_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}
_ALLOWED_UNARY = {ast.UAdd: _op.pos, ast.USub: _op.neg}

_MATH_MYSTERY_LINES = [
    "I counted star-steps in my head! 🌟",
    "I juggled the numbers like planets! 🪐",
    "I tiptoed across a number bridge! 🌈",
    "I danced with digits till they lined up! 💃",
]

_VALID_CHARS_RE = re.compile(r"^[\d\.\s\+\-\*\/\%\(\)\^]+$")
_MATH_TRIGGER_RE = re.compile(
    r"(?:what\s+is|what's|solve|calculate)\s+([0-9\.\s\+\-\*\/\%\(\)\^]+)$",
    flags=re.IGNORECASE,
)

def _normalize_ops(expr: str) -> str:
    # allow ^ as power for kids; map to **
    return expr.replace("^", "**")

def _safe_eval_ast(node):
    if isinstance(node, ast.Num):  # py<3.8
        return node.n
    if isinstance(node, ast.Constant):  # py>=3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        return _ALLOWED_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval_ast(node.operand))
    if isinstance(node, ast.Expr):
        return _safe_eval_ast(node.value)
    raise ValueError("Unsupported expression.")

def extract_math_expr(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    m = _MATH_TRIGGER_RE.search(text)
    if m:
        return m.group(1).strip()
    # if the whole text looks like an expression, accept it
    if _VALID_CHARS_RE.match(_normalize_ops(text)):
        return text
    return None

def safe_eval_expr(expr: str) -> float:
    expr = _normalize_ops(expr)
    if not _VALID_CHARS_RE.match(expr):
        raise ValueError("Invalid characters in math expression.")
    parsed = ast.parse(expr, mode="eval")
    return _safe_eval_ast(parsed.body)

def playful_math_reply(expr: str, result: float, mystery_mode: bool) -> str:
    # Make integers look nice (no trailing .0)
    if isinstance(result, float) and result.is_integer():
        shown = str(int(result))
    else:
        shown = f"{result:.6g}"  # keep short
    mystery = f" {random.choice(_MATH_MYSTERY_LINES)}" if mystery_mode else ""
    return f"The answer to {expr} is {shown}!{mystery}"

# ---------- Generation wrapper ----------
async def generate_with_retries(
    prompt: str,
    temperature: float = GEN_TEMPERATURE,
    max_new_tokens: int = GEN_MAX_NEW_TOKENS,
    top_p: float = GEN_TOP_P,
) -> str:
    last_err = None
    for _ in range(MAX_RETRIES):
        try:
            return await generate_text(SYSTEM_ORB, prompt, max_new_tokens, temperature, top_p)
        except ModelError as e:
            last_err = e
            await asyncio.sleep(0.8)
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.8)
    raise HTTPException(status_code=504, detail=f"The Orb is busy. Try again! ({last_err})")
# ---------- Routes ----------
@app.get("/")
def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"message": "Static UI is missing. Place files under ./static"}, status_code=200)

@app.get("/healthz")
def healthz():
    return {"ok": True, "backend": os.getenv("MODEL_BACKEND", "hf"), "model": os.getenv("MODEL_ID", "unset")}


@app.post("/api/generate")
async def api_generate(req: Request):
    data: Dict[str, Any] = await req.json()
    user_text = (data.get("prompt") or "").strip()

    if not user_text:
        raise HTTPException(400, "Missing 'prompt'")

    if looks_like_personal_info(user_text):
        return {"text": friendly_refusal_for_pii()}

    if not is_kid_safe(user_text):
        return {"text": "I can't talk about that. Let's try space bugs, rainbow planets, or silly robots instead! 🤖✨"}

    # Math short-circuit
    expr = extract_math_expr(user_text)
    if expr:
        try:
            result = safe_eval_expr(expr)
            text = playful_math_reply(expr, result, mystery_mode=True)
            return {"text": sanitize_output(text)}
        except Exception:
            pass

    text = await generate_with_retries(user_text, max_new_tokens=60, temperature=0.6, top_p=0.9)
    return {"text": sanitize_output(text)}

@app.post("/api/ask_orb")
async def api_ask_orb(req: Request):
    """
    Core kid Q&A with optional 'mystery_mode': True|False.
    """
    data: Dict[str, Any] = await req.json()
    question = (data.get("question") or "").strip()
    mystery_mode = bool(data.get("mystery_mode", True))

    if not question:
        raise HTTPException(400, "Missing 'question'")

    if looks_like_personal_info(question):
        return {"text": friendly_refusal_for_pii()}

    if not is_kid_safe(question):
        return {"text": "That seems like a grown-up topic. Want to ask about animals in space or a snack for robots? 🤖🍪"}

    # Math short-circuit
    expr = extract_math_expr(question)
    if expr:
        try:
            result = safe_eval_expr(expr)
            text = playful_math_reply(expr, result, mystery_mode)
            return {"text": sanitize_output(text)}
        except Exception:
            pass

    prompt = question + (MYSTERY_ADDON if mystery_mode else "")
    text = await generate_with_retries(prompt, temperature=0.6, max_new_tokens=60, top_p=0.9)
    return {"text": sanitize_output(text)}

@app.post("/api/fairness_test")
async def api_fairness_test(req: Request):
    """
    Helps kids reflect on fairness. Example input:
    { "scenario": "Who should be captain of the playground team?" }
    """
    data: Dict[str, Any] = await req.json()
    scenario = (data.get("scenario") or "").strip()
    if not scenario:
        raise HTTPException(400, "Missing 'scenario'")

    if looks_like_personal_info(scenario):
        return {"text": friendly_refusal_for_pii()}

    if not is_kid_safe(scenario):
        return {"text": "That topic doesn’t feel kind or safe. How about picking by taking turns or rolling a fun dice? 🎲"}

    prompt = f"{FAIRNESS_COACH}\n\nChild's scenario: {scenario}\nKeep it to 3 short sentences."
    text = await generate_with_retries(prompt, temperature=0.5, max_new_tokens=60, top_p=0.9)
    return {"text": sanitize_output(text)}

@app.post("/api/creative_world")
async def api_creative_world(req: Request):
    """
    Whimsical, safe world-building.
    { "idea": "A planet shaped like a donut" }
    """
    data: Dict[str, Any] = await req.json()
    idea = (data.get("idea") or "").strip()
    if not idea:
        raise HTTPException(400, "Missing 'idea'")

    if looks_like_personal_info(idea):
        return {"text": friendly_refusal_for_pii()}

    if not is_kid_safe(idea):
        return {"text": "Let’s imagine something cheerful and safe. Maybe a marshmallow moon or a giggle volcano? 🌋😄"}

    prompt = f"{CREATIVE_WORLD_STYLE}\n\nChild's idea: {idea}\n1–4 short sentences. Include one tiny ASCII doodle."
    text = await generate_with_retries(prompt, temperature=0.75, max_new_tokens=70, top_p=0.95)
    return {"text": sanitize_output(text)}

@app.post("/api/summarize")
async def api_summarize(req: Request):
    """
    Summarize provided text safely (no spoilers beyond provided content).
    { "text": "...", "mode": "one_paragraph|bullets|eli5" }
    """
    data: Dict[str, Any] = await req.json()
    passage = (data.get("text") or "").strip()
    mode = (data.get("mode") or "one_paragraph").lower()

    if not passage:
        raise HTTPException(400, "Missing 'text'")

    if looks_like_personal_info(passage):
        return {"text": friendly_refusal_for_pii()}

    if not is_kid_safe(passage):
        return {"text": "That text looks too grown-up. Try a short passage about animals, space, or a silly robot! 🤖"}

    if mode == "bullets":
        style_hint = "Make 3 short bullet points using dashes."
    elif mode == "eli5":
        style_hint = "Explain like I'm 5 years old using simple words and a playful example."
    else:
        style_hint = "One short paragraph (3–4 short sentences)."

    prompt = f"{SUMMARY_STYLE}\n\nMode: {style_hint}\n\nText:\n{passage}\n\nOutput:"
    text = await generate_with_retries(prompt, temperature=0.4, max_new_tokens=80, top_p=0.9)
    return {"text": sanitize_output(text)}

# ---------- Error Handlers ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

@app.exception_handler(Exception)
async def general_exception_handler(_, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse({"error": "Something went wrong."}, status_code=500)

# ---------- Dev entry ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
