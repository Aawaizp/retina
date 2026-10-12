"""RETINA LLM helper: talk to Groq, Gemini, Hugging Face or Ollama with ONE function.

All four speak the same "OpenAI-compatible" chat format, so only the web address, the key
and the model name change. API keys are read from environment variables - never put a key
in the code (it would end up in screenshots, GitHub, ...).

"auto" (the default) uses a CHAIN: Groq first (best), then Gemini, then Ollama on your PC.
If one has no key, hits its free limit or cannot be reached, the next one answers - and the busy
one rests for a minute. Change the order:  $env:RETINA_LLM_CHAIN = "gemini,groq,ollama"
Cerebras is available but NOT in the default chain: in Oct 2026 its free trial asks for a verified
payment method (RETINA rule: nothing that needs a card). Opt in:  $env:RETINA_LLM_CHAIN = "cerebras,groq,..."

Rate limiter: before every request the PACER waits until the provider's per-minute limits
(requests and tokens) have room - so free tiers are not hit in the first place (retries on 429
stay as a safety net). Daily use is counted in data/cache/llm_usage.json; a provider whose free
day is used up is skipped until tomorrow (UTC). Limits: LIMITS below, or override one provider:
  $env:RETINA_LIMITS_GROQ = "rpm=30,tpm=8000,rpd=1000,tpd=200000"

Test it:   python -m retina.engine.llm               (auto: asks one short question)
           python -m retina.engine.llm groq          (one provider)
           python -m retina.engine.llm groq models   (lists the models you can use)
"""

import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

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
    "cerebras": {"url": "https://api.cerebras.ai/v1", "key": "CEREBRAS_API_KEY",   # opt-in (see above)
                 "model": "gpt-oss-120b"},
}
DEFAULT_PROVIDER = os.environ.get("RETINA_LLM", "auto")   # change without editing code
CHAIN = [p.strip() for p in os.environ.get("RETINA_LLM_CHAIN", "groq,gemini,ollama").split(",") if p.strip()]
RESTING = {}                 # provider -> time it may be used again (after a limit or a dropped line)
LAST_USED = {"provider": None, "model": None}     # who answered the last chat() call

# Free-tier limits per provider (default model). rpm/tpm = per minute, rpd/tpd = per day; None = no limit.
# groq: console.groq.com/docs/rate-limits (gpt-oss-120b, checked 12 Oct 2026) - PUBLIC_STATISTIC
# cerebras: inference-docs.cerebras.ai/support/rate-limits (free trial, checked 12 Oct 2026) - PUBLIC_STATISTIC
# gemini: Google shows limits only inside AI Studio; these are cautious guesses - ESTIMATED
# ollama: your own PC, no limits
LIMITS = {
    "groq": {"rpm": 30, "tpm": 8_000, "rpd": 1_000, "tpd": 200_000},
    "gemini": {"rpm": 10, "tpm": 250_000, "rpd": 250, "tpd": None},
    "cerebras": {"rpm": 5, "tpm": 30_000, "rpd": None, "tpd": 1_000_000},
    "hf": {"rpm": 10, "tpm": None, "rpd": None, "tpd": None},
    "ollama": {"rpm": None, "tpm": None, "rpd": None, "tpd": None},
}
USAGE_FILE = Path(__file__).resolve().parents[2] / "data" / "cache" / "llm_usage.json"


def limits(provider):
    """LIMITS for a provider, with $env:RETINA_LIMITS_<PROVIDER> = "rpm=30,tpm=8000" applied on top."""
    out = dict(LIMITS.get(provider, {"rpm": None, "tpm": None, "rpd": None, "tpd": None}))
    for part in os.environ.get(f"RETINA_LIMITS_{provider.upper()}", "").split(","):
        name, _, value = part.partition("=")
        if name.strip() in out and value.strip():
            out[name.strip()] = None if value.strip().lower() == "none" else int(value)
    return out


def estimate_tokens(messages, reply_tokens=800):
    """Rough token count of a request: ~4 characters per token, plus room for the reply."""
    return sum(len(m.get("content") or "") for m in messages) // 4 + reply_tokens


class Pacer:
    """Keeps each provider under its per-minute limits (waits BEFORE sending) and counts daily use.
    The per-minute window lives in memory; daily totals are saved in USAGE_FILE (UTC days)."""

    def __init__(self, usage_file=USAGE_FILE, clock=time.time, sleep=time.sleep):
        self.file, self.clock, self.sleep = Path(usage_file), clock, sleep
        self.window = {}                                   # provider -> deque of (time, tokens)

    def today(self):
        return datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y-%m-%d")

    def _usage(self):
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        return data if data.get("day") == self.today() else {"day": self.today(), "providers": {}}

    def used_today(self, provider):
        u = self._usage()["providers"].get(provider, {})
        return u.get("requests", 0), u.get("tokens", 0)

    def left_today(self, provider):
        """(requests, tokens) still free today; None = no daily limit."""
        lim, (req, tok) = limits(provider), self.used_today(provider)
        return (None if lim["rpd"] is None else max(lim["rpd"] - req, 0),
                None if lim["tpd"] is None else max(lim["tpd"] - tok, 0))

    def wait(self, provider, tokens, say=print):
        """Block until one more request of `tokens` fits this minute. Raises if today's limit is used up."""
        lim = limits(provider)
        req_left, tok_left = self.left_today(provider)
        if (req_left is not None and req_left < 1) or (tok_left is not None and tok_left < tokens):
            raise RuntimeError(f"{provider}: daily limit reached (free tier) - try again tomorrow")
        window = self.window.setdefault(provider, deque())
        told = False
        while True:
            now = self.clock()
            while window and window[0][0] <= now - 60:
                window.popleft()
            busy_req = lim["rpm"] is not None and len(window) + 1 > lim["rpm"]
            busy_tok = lim["tpm"] is not None and window and sum(t for _, t in window) + tokens > lim["tpm"]
            if not (busy_req or busy_tok):
                window.append((now, tokens))
                return
            pause = max(window[0][0] + 60 - now, 0.5)
            if not told:
                say(f"  ({provider}: pacing under its free per-minute limit - {pause:.0f}s)")
                told = True
            self.sleep(pause)

    def spent(self, provider, tokens):
        """After a request: replace the guess in this minute's window by the real count, add to today."""
        window = self.window.get(provider)
        if window:
            t, _ = window.pop()
            window.append((t, tokens))
        data = self._usage()
        u = data["providers"].setdefault(provider, {"requests": 0, "tokens": 0})
        u["requests"] += 1
        u["tokens"] += int(tokens)
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            self.file.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except OSError:
            pass                                           # counting is a help, never a reason to stop


PACER = Pacer()


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
            if any(k in text for k in ("429", "Could not reach", "error 5", "error 403", "error 401",
                                       "daily limit")):
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
    guess = estimate_tokens(body["messages"])
    PACER.wait(provider, guess)                               # stay under the per-minute limits
    try:
        data = send("POST", url + "/chat/completions", headers, provider, patient, json=body)
    except RuntimeError as error:
        if "reasoning_effort" not in body or "error 400" not in str(error):
            raise
        body.pop("reasoning_effort")                      # this provider does not accept it
        data = send("POST", url + "/chat/completions", headers, provider, patient, json=body)
    PACER.spent(provider, (data.get("usage") or {}).get("total_tokens") or guess)
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