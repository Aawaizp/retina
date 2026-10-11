"""RETINA LLM helper: talk to Groq, Gemini, Hugging Face or Ollama with ONE function.

All four speak the same "OpenAI-compatible" chat format, so only the web address, the key
and the model name change. API keys are read from environment variables - never put a key
in the code (it would end up in screenshots, GitHub, ...).

"auto" (the default) uses a CHAIN: Groq first (best), then Gemini, then Ollama on your PC.
If one has no key, hits its free limit or cannot be reached, the next one answers - and the busy
one rests for a minute. Change the order:  $env:RETINA_LLM_CHAIN = "gemini,groq,ollama"

Test it:   python simulation\\llm.py               (auto: asks one short question)
           python simulation\\llm.py groq          (one provider)
           python simulation\\llm.py groq models   (lists the models you can use)
"""

import json
import os
import sys
import time

import requests

PROVIDERS = {
    # name: web address, environment variable holding the key, default model
    "groq": {"url": "https://api.groq.com/openai/v1", "key": "GROQ_API_KEY",
             "model": "openai/gpt-oss-120b"},
    "gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/openai", "key": "GEMINI_API_KEY",
               "model": "gemini-3.5-flash-lite"},
    "hf": {"url": "https://router.huggingface.co/v1", "key": "HF_TOKEN",
           "model": "openai/gpt-oss-120b"},
    "ollama": {"url": "http://localhost:11434/v1", "key": None,          # runs on your own PC, free
               "model": "llama3.2:3b"},                                   # 2 GB; smaller: qwen3:1.7b (1.4 GB)
}
DEFAULT_PROVIDER = os.environ.get("RETINA_LLM", "auto")   # change without editing code
CHAIN = [p.strip() for p in os.environ.get("RETINA_LLM_CHAIN", "groq,gemini,ollama").split(",") if p.strip()]
RESTING = {}                 # provider -> time it may be used again (after a limit or a dropped line)
LAST_USED = {"provider": None, "model": None}     # who answered the last chat() call


def connect(provider):
    """Web address, headers (with the key) and default model for a provider."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}")
    p = PROVIDERS[provider]
    headers = {"Content-Type": "application/json"}
    if p["key"]:
        key = os.environ.get(p["key"])
        if not key:
            raise RuntimeError(f"No key for {provider}. In PowerShell run:  $env:{p['key']} = \"your-key\"")
        headers["Authorization"] = f"Bearer {key}"
    return p["url"], headers, p["model"]


def send(method, url, headers, provider, patient=True, **kwargs):
    """One web request with friendly errors. Free tiers limit requests per minute (HTTP 429):
    then wait a little and try again (patient=False: give up at once, the chain moves on)."""
    for attempt in range(4):
        try:
            r = requests.request(method, url, headers=headers, timeout=300 if provider == "ollama" else 120, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as error:
            if provider != "ollama" and attempt < (2 if patient else 1):   # the line often drops: retry
                wait = 5 * (attempt + 1)
                print(f"  ({provider}: connection problem, {type(error).__name__} - trying again in {wait}s)")
                time.sleep(wait)
                continue
            hint = " Is Ollama running? (install from ollama.com, then: ollama pull llama3.2:3b)" \
                if provider == "ollama" else " Check your internet connection, or try another provider (--provider gemini)."
            raise RuntimeError(f"Could not reach {provider} ({type(error).__name__}).{hint}")
        if r.status_code in (429, 500, 502, 503) and attempt < (3 if patient else 0):   # limit or overloaded
            wait = 5 * (attempt + 1)
            print(f"  ({provider} {'rate limit' if r.status_code == 429 else 'overloaded'} - waiting {wait}s)")
            time.sleep(wait)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"{provider} answered with error {r.status_code}: {r.text[:300]}")
        return r.json()


def ready(provider):
    """Has a key (or needs none) and is not resting after a limit."""
    p = PROVIDERS.get(provider)
    return p is not None and (not p["key"] or os.environ.get(p["key"])) and RESTING.get(provider, 0) <= time.time()


def chat_auto(messages, temperature=0.2, reasoning=None):
    """Try the providers in CHAIN order; a busy or missing one is skipped."""
    problems = []
    for provider in CHAIN:
        if not ready(provider):
            problems.append(f"{provider}: no key" if RESTING.get(provider, 0) <= time.time() else f"{provider}: resting")
            continue
        try:
            return chat(messages, provider, None, temperature, reasoning, patient=False)
        except RuntimeError as error:
            text = str(error)
            if any(k in text for k in ("429", "Could not reach", "error 5", "error 403", "error 401")):
                RESTING[provider] = time.time() + 60          # let it rest, use the next one
                problems.append(f"{provider}: {text[:80]}")
                print(f"  ({provider} busy or unavailable - switching to the next provider)")
                continue
            raise
    raise RuntimeError("No LLM provider could answer (" + "; ".join(problems) + "). "
                       "Set GROQ_API_KEY or GEMINI_API_KEY, or start Ollama.")


def chat(messages, provider=None, model=None, temperature=0.2, reasoning=None, patient=True):
    """Send a conversation, get the reply text.
    messages = [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    reasoning = "low" asks "thinking" models (gpt-oss) to think less: fewer tokens, so free-tier
    token-per-minute limits are hit less often. Ignored by models that do not support it."""
    provider = provider or DEFAULT_PROVIDER
    if provider == "auto":
        return chat_auto(messages, temperature, reasoning)
    url, headers, default_model = connect(provider)
    model = model or default_model
    body = {"model": model, "messages": messages, "temperature": temperature}
    if reasoning and "gpt-oss" in model and provider in ("groq", "hf", "ollama"):
        body["reasoning_effort"] = reasoning
    if reasoning == "low" and "qwen3" in model.lower():      # small Qwen3 models: skip "thinking" = faster
        body["messages"] = messages[:-1] + [{**messages[-1], "content": messages[-1]["content"] + "\n/no_think"}]
    try:
        data = send("POST", url + "/chat/completions", headers, provider, patient, json=body)
    except RuntimeError as error:
        if "reasoning_effort" not in body or "error 400" not in str(error):
            raise
        body.pop("reasoning_effort")                      # this provider does not accept it
        data = send("POST", url + "/chat/completions", headers, provider, patient, json=body)
    LAST_USED.update(provider=provider, model=model)
    message = data["choices"][0]["message"]
    text = (message.get("content") or "").strip()
    if not text and message.get("tool_calls"):        # some models use their own tool-call format:
        call = message["tool_calls"][0]["function"]  # turn it into our {"tool": ..., "args": ...}
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        text = json.dumps({"tool": call.get("name"), "args": args})
    return text

def json_from(text, kind="["):
    """The first JSON list (or object) in a reply - models sometimes add text or a second block after it."""
    start = text.find(kind)
    if start < 0:
        raise ValueError("no JSON in the reply")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def list_models(provider):
    """Model names this provider offers right now (they change often)."""
    url, headers, _ = connect(provider)
    data = send("GET", url + "/models", headers, provider)
    return sorted(m["id"] for m in data["data"])


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROVIDER
    try:
        if len(sys.argv) > 2 and sys.argv[2] == "models":
            models = list_models(name)
            print(f"{name}: {len(models)} models")
            for m in models[:40]:
                print("  ", m)
        else:
            start = time.time()
            reply = chat([{"role": "user", "content":
                           "In two sentences: why do supermarkets run price promotions?"}], name)
            print(f"{LAST_USED['provider']} ({LAST_USED['model']}) answered in {time.time() - start:.1f}s:\n{reply}")
    except (RuntimeError, ValueError) as error:
        print("Problem:", error)