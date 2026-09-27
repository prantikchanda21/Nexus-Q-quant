"""Groq chat completion client with a two-key fallback chain.

Request routing:

    1. GROQ_API_KEY        → primary Groq key
    2. GROQ_API_KEY_FALLBACK → secondary Groq key (used if the primary is
       missing, rate-limited 429, invalid 401, or erroring 5xx)
    3. local summarizer    → deterministic offline answer built from the
       assessment ledger (never fails, clearly labeled in the response)

Keys are read from environment variables (optionally via a .env file in the
backend directory). ``groq`` is imported lazily so the app boots fine without
the package installed.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

try:  # optional .env support without adding a hard dependency
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

DEFAULT_MODEL = "openai/gpt-oss-120b"          # Groq production flagship (~500 t/s)
DEFAULT_FALLBACK_MODEL = "openai/gpt-oss-20b"  # fastest/cheapest production model
REQUEST_TIMEOUT_S = 30.0
MAX_CONTEXT_CHARS = 7000
# gpt-oss models are reasoners: give them headroom and keep effort low so the
# visible answer never gets starved by hidden chain-of-thought tokens.
MAX_TOKENS = 1600
REASONING_EFFORT = "low"

# Provider-level retryable conditions.
_RETRYABLE = {"rate_limit_exceeded", "service_unavailable", "api_error", "timeout"}


def _models() -> List[str]:
    """Ordered model chain: env-overridable, de-duplicated."""
    primary = os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_MODEL
    secondary = os.environ.get("GROQ_MODEL_FALLBACK", "").strip() or DEFAULT_FALLBACK_MODEL
    models: List[str] = []
    for model in (primary, secondary):
        if model and model not in models:
            models.append(model)
    return models


class GroqError(RuntimeError):
    """Raised when every Groq key has been exhausted."""


def _load_groq():
    """Import and return the groq module, or None when unavailable."""
    try:
        import groq  # type: ignore

        return groq
    except ImportError:
        return None


def _candidate_keys() -> List[str]:
    """Ordered, de-duplicated list of Groq API keys to try."""
    primary = os.environ.get("GROQ_API_KEY", "").strip()
    fallback = os.environ.get("GROQ_API_KEY_FALLBACK", "").strip()
    keys: List[str] = []
    for key in (primary, fallback):
        if key and key not in keys:
            keys.append(key)
    return keys


class GroqChatClient:
    """Chat-completions client with primary → fallback key resilience."""

    def __init__(self) -> None:
        self._groq = _load_groq()
        # One client per key; built lazily and cached by key string.
        self._clients: Dict[str, Any] = {}
        self._last_key_used: Optional[str] = None

    @property
    def available(self) -> bool:
        """True when the groq package and at least one API key are present."""
        return self._groq is not None and bool(_candidate_keys())

    @property
    def n_keys(self) -> int:
        """Number of configured Groq keys."""
        return len(_candidate_keys())

    @property
    def last_key_used(self) -> Optional[str]:
        """Label of the key that served the last successful call."""
        return self._last_key_used

    def _client_for(self, api_key: str) -> Any:
        client = self._clients.get(api_key)
        if client is None:
            client = self._groq.Groq(api_key=api_key)
            self._clients[api_key] = client
        return client

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 900,
    ) -> Dict[str, Any]:
        """Run a chat completion across the key × model resilience chain.

        Tries every configured API key with the primary model first, then the
        fast secondary model — so a per-model rate limit degrades to the
        smaller model before burning the second key.

        Args:
            messages: OpenAI-style message dicts ({role, content}).
            temperature: Sampling temperature.
            max_tokens: Response budget.

        Returns:
            {answer, model, key_label, usage} on success.

        Raises:
            GroqError: If all key/model combos fail (the route then uses the
                local responder).
        """
        if self._groq is None:
            raise GroqError("groq package not installed (pip install groq)")

        keys = _candidate_keys()
        if not keys:
            raise GroqError("no GROQ_API_KEY / GROQ_API_KEY_FALLBACK configured")
        models = _models()

        errors: List[str] = []
        for index, api_key in enumerate(keys):
            label = "primary" if index == 0 else "fallback"
            for model in models:
                # gpt-oss accepts reasoning_effort; older/other models reject it.
                kwargs_variants = [
                    {"reasoning_effort": REASONING_EFFORT, "max_tokens": max(max_tokens, MAX_TOKENS)},
                    {"max_tokens": max_tokens},
                ]
                last_variant_error: Optional[Exception] = None
                for kwargs in kwargs_variants:
                    if attempt_no := getattr(self, "_skip_reasoning_param", False):
                        kwargs = {k: v for k, v in kwargs.items() if k != "reasoning_effort"}
                    for attempt in (1, 2):  # one transient retry per combo
                        try:
                            started = time.perf_counter()
                            response = self._client_for(api_key).chat.completions.create(
                                model=model,
                                messages=messages,  # type: ignore[arg-type]
                                temperature=temperature,
                                timeout=REQUEST_TIMEOUT_S,
                                **kwargs,
                            )
                            self._last_key_used = label
                            raw_answer = (response.choices[0].message.content or "").strip()
                            # A reasoner can exhaust max_tokens on hidden reasoning
                            # and return an empty answer — treat that as a retryable
                            # model-level failure and drop to the next variant.
                            if not raw_answer:
                                last_variant_error = GroqError(f"{model} returned empty answer (reasoning exhausted budget)")
                                break
                            usage = getattr(response, "usage", None)
                            return {
                                "answer": raw_answer,
                                "model": model,
                                "key_label": label,
                                "latency_ms": (time.perf_counter() - started) * 1000.0,
                                "usage": {
                                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                                    "completion_tokens": getattr(usage, "completion_tokens", None),
                                },
                            }
                        except TypeError as exc:
                            # SDK lacks reasoning_effort → remember and drop it.
                            self._skip_reasoning_param = True
                            last_variant_error = exc
                            break
                        except Exception as exc:  # noqa: BLE001 — classified below
                            last_variant_error = exc
                            status = getattr(exc, "status_code", None)
                            body = getattr(exc, "body", None)
                            err_code = None
                            if isinstance(body, dict):
                                error_field = body.get("error", {})
                                if isinstance(error_field, dict):
                                    err_code = error_field.get("code")
                                elif isinstance(error_field, str):
                                    err_code = error_field

                            retryable = status in (401, 429, 500, 502, 503) or err_code in _RETRYABLE
                            errors.append(
                                f"{label}/{model}: {type(exc).__name__} {status or ''} {err_code or ''} {exc}"
                            )

                            if not retryable or attempt >= 2:
                                break
                            time.sleep(0.6)
                if isinstance(last_variant_error, GroqError) and "empty answer" in str(last_variant_error):
                    errors.append(f"{label}/{model}: {last_variant_error}")

        raise GroqError(" | ".join(errors))


# Module-level singleton used by the chat route.
CLIENT = GroqChatClient()
