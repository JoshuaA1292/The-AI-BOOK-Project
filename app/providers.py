# app/providers.py
import os, httpx
from typing import List, Dict

BACKEND = os.getenv("MODEL_BACKEND", "hf").lower()      # we'll use "hf"
MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
TIMEOUT = float(os.getenv("GEN_TIMEOUT_S", "25"))

class ModelError(RuntimeError): ...
def _messages_to_instruct(system: str, user: str) -> str:
    # Simple instruct format that works well with Mistral Instruct
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n{user} [/INST]"

async def generate_text(system: str, prompt: str,
                        max_new_tokens: int = 64, temperature: float = 0.6, top_p: float = 0.9) -> str:
    if BACKEND != "hf":
        raise ModelError(f"Unsupported backend {BACKEND}; set MODEL_BACKEND=hf")

    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise ModelError("Missing HF_API_TOKEN")

    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    instruct = _messages_to_instruct(system, prompt)
    payload = {
        "inputs": instruct,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False,
        }
    }
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise ModelError(f"HF {r.status_code}: {r.text[:300]}")
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "error" in data:
            raise ModelError(f"HF: {data['error']}")
        raise ModelError("HF response shape not recognized")
