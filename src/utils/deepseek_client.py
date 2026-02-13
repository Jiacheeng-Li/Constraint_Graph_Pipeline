import os
import time
from typing import Any, Dict, List, Optional

import requests


DEFAULT_ENDPOINT = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-dbb4d81266e34933896235c07d3ad543")

LLM_CALL_EVENTS: List[Dict[str, Any]] = []


class DeepSeekError(Exception):
    pass


def _log_llm_event(success: bool,
                   model: str,
                   endpoint: str,
                   duration_sec: Optional[float],
                   error: Optional[str] = None) -> None:
    LLM_CALL_EVENTS.append({
        "success": success,
        "model": model,
        "endpoint": endpoint,
        "duration_sec": duration_sec,
        "error": error,
        "timestamp": time.time(),
    })


def call_chat_completions(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: int = 120,
    retries: int = 2,
    retry_backoff_sec: float = 0.8,
) -> str:
    """
    Minimal DeepSeek chat.completions wrapper.

    Returns the assistant content string. Raises DeepSeekError on failure.
    """

    used_endpoint = endpoint or DEFAULT_ENDPOINT
    used_model = model or DEFAULT_MODEL
    used_api_key = (api_key or DEFAULT_API_KEY).strip()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {used_api_key}",
    }

    payload: Dict[str, Any] = {
        "model": used_model,
        "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err: Optional[Exception] = None
    total_start = time.time()
    for attempt in range(retries + 1):
        try:
            attempt_start = time.time()
            resp = requests.post(used_endpoint, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise DeepSeekError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise DeepSeekError("Empty choices in response")
            message = (choices[0] or {}).get("message") or {}
            content = (message or {}).get("content")
            if not content:
                raise DeepSeekError("Empty content in first choice")
            _log_llm_event(
                success=True,
                model=used_model,
                endpoint=used_endpoint,
                duration_sec=time.time() - total_start,
            )
            return content
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(retry_backoff_sec * (attempt + 1))
            else:
                break

    _log_llm_event(
        success=False,
        model=used_model,
        endpoint=used_endpoint,
        duration_sec=time.time() - total_start,
        error=str(last_err) if last_err else "Unknown error",
    )
    raise DeepSeekError(str(last_err) if last_err else "Unknown DeepSeek error")
