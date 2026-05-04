"""OpenAI-compatible client wrapper for the frozen Actor (vLLM Qwen2.5-7B at port 8001).

Used by every module of the compound program (solve, diagnose, advise, revise).
The Advisor and Actor share the same backing model in our setup; what differs
is the prompt (theta) GEPA evolves.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from openai import OpenAI


@dataclass(frozen=True)
class ActorConfig:
    base_url: str = os.environ.get("RAE_ACTOR_BASE_URL", "http://localhost:8001/v1")
    api_key: str = os.environ.get("RAE_ACTOR_API_KEY", "sk-capr-actor")
    model: str = os.environ.get("RAE_ACTOR_MODEL", "Qwen/Qwen2.5-72B-Instruct-AWQ")
    temperature: float = 0.0
    max_tokens: int = (
        2800  # Tax problems are ~7k tokens; leave room within mml=16384 context.
    )
    timeout_s: float = 600.0  # 72B-AWQ generation is slower; generous timeout.
    max_retries: int = 3


class ActorClient:
    """Thin wrapper around openai.OpenAI for chat.completions.

    Stateless and thread-safe: a single client may be shared across worker
    threads inside GEPA's parallel evaluator.
    """

    def __init__(self, cfg: Optional[ActorConfig] = None):
        self.cfg = cfg or ActorConfig()
        self._client = OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            timeout=self.cfg.timeout_s,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                rsp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature
                    if temperature is None
                    else temperature,
                    max_tokens=self.cfg.max_tokens
                    if max_tokens is None
                    else max_tokens,
                    seed=seed,
                )
                return rsp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError(
            f"Actor call failed after {self.cfg.max_retries} attempts: {last_err}"
        )
