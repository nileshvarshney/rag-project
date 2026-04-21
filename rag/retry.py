"""
retry.py — Tenacity retry configuration for Ollama LLM calls.

Why retry: local Ollama can transiently fail when the process is busy
loading a model, the GPU is swapping, or the OS rejects a connection
under load. These failures resolve on their own in seconds — a simple
exponential back-off recovers without user intervention.

What we retry: only transient transport-layer errors (connect refused,
read timeout, server disconnected). Logic errors like OutputParserException
or ValueError are not retried — they won't self-heal.

Strategy (4 attempts total, 3 retries):
  attempt 1 → fail → wait 2–3 s  (base + jitter)
  attempt 2 → fail → wait 4–5 s
  attempt 3 → fail → wait 8–9 s
  attempt 4 → fail → reraise original exception

Jitter: random ±1 s added to each wait so concurrent retries from
multiple workers don't all hit Ollama at the same moment (thundering herd).
"""

import logging

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

logger = logging.getLogger(__name__)

# Transport-level errors that are worth retrying.
_RETRIABLE = (
    httpx.ConnectError,        # Ollama not yet reachable (model loading)
    httpx.ReadTimeout,         # Ollama took too long to return a token
    httpx.RemoteProtocolError, # Server closed connection mid-stream
    ConnectionError,           # OS-level: connection refused / reset
    TimeoutError,              # Generic Python timeout
)

# Exponential backoff (2→4→8 s) plus up to 1 s of random jitter.
# Jitter prevents concurrent workers from all retrying at the same instant.
_backoff = wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 1)

# Reusable decorator — apply to any function that makes a single LLM call.
ollama_retry = retry(
    retry=retry_if_exception_type(_RETRIABLE),
    wait=_backoff,
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,  # after 4 attempts, re-raise the original exception
)


@ollama_retry
def retry_invoke(chain, inputs: dict):
    """Call chain.invoke(inputs) with automatic retry on transient errors.

    Each call site gets its own independent retry counter — wrapping
    individual chain calls (not whole functions) keeps retry scope tight:
    a single chunk's grader call failing won't retry the entire pipeline.

    Args:
        chain:  Any LangChain Runnable that supports .invoke().
        inputs: The dict of inputs to pass to the chain.

    Returns:
        The chain's output (usually a string for StrOutputParser chains).

    Raises:
        The original exception after 4 failed attempts.
    """
    return chain.invoke(inputs)


@ollama_retry
async def async_retry_invoke(chain, inputs: dict):
    """Async equivalent of retry_invoke — uses chain.ainvoke().

    Tenacity's @retry decorator supports async functions natively: it awaits
    the coroutine and applies the same exponential back-off on failure.

    Use this in async node functions so the event loop is never blocked
    waiting for an Ollama response.
    """
    return await chain.ainvoke(inputs)
