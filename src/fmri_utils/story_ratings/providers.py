from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import ProviderConfig

USAGE_LIMIT_PATTERN = re.compile(
    r"(usage|rate)[ _-]?limit|limit (reached|resets|will reset)|hit your (usage )?limit|insufficient_quota",
    re.IGNORECASE,
)

# Published list prices per million tokens, for cost estimates in run
# summaries. Unknown models report no cost rather than a guess.
MODEL_PRICES_USD_PER_MTOK: Dict[str, Dict[str, float]] = {
    "claude-opus-5": {"input": 5.0, "output": 25.0},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    "gpt-5.6-terra": {"input": 2.0, "output": 12.0},
    "gpt-5.6-sol": {"input": 4.0, "output": 20.0},
    "gpt-5-mini": {"input": 0.25, "output": 2.0},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
}


class UsageLimitError(RuntimeError):
    """The account or key hit its usage limit; later calls would fail too."""


@dataclass
class Usage:
    """Token counts and wall time for one model call."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    duration_ms: float = 0.0
    calls: int = 1

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            duration_ms=self.duration_ms + other.duration_ms,
            calls=self.calls + other.calls,
        )

    def cost_usd(self, model: str) -> Optional[float]:
        prices = MODEL_PRICES_USD_PER_MTOK.get(model)
        if prices is None:
            return None
        return self.input_tokens / 1e6 * prices["input"] + self.output_tokens / 1e6 * prices["output"]

    def to_dict(self, model: str = "") -> Dict[str, Any]:
        payload = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "duration_ms": round(self.duration_ms),
            "calls": self.calls,
        }
        if model:
            payload["estimated_cost_usd"] = self.cost_usd(model)
        return payload


def read_api_key(name: str) -> str:
    """Read an API key from the environment, falling back to the Windows user store.

    A process started before the key was set inherits an environment without
    it, which is a common and confusing failure on Windows.
    """
    key = os.environ.get(name)
    if not key and os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
                key = winreg.QueryValueEx(handle, name)[0]
        except (FileNotFoundError, OSError):
            key = None
    if not key:
        raise RuntimeError(f"{name} is not set")
    return key


def _anthropic_schema(schema: Any) -> Any:
    """Anthropic structured outputs reject minimum/maximum, so use enums."""
    if isinstance(schema, dict):
        if schema.get("type") == "integer" and "minimum" in schema and "maximum" in schema:
            return {"type": "integer", "enum": list(range(schema["minimum"], schema["maximum"] + 1))}
        return {key: _anthropic_schema(value) for key, value in schema.items()}
    if isinstance(schema, list):
        return [_anthropic_schema(item) for item in schema]
    return schema


def call_anthropic(prompt: str, schema: Mapping[str, Any], config: ProviderConfig) -> tuple[Dict[str, Any], Usage]:
    """One rating request to the Anthropic Messages API."""
    import anthropic

    client = anthropic.Anthropic(api_key=read_api_key("ANTHROPIC_API_KEY"), timeout=float(config.timeout_seconds))
    request: Dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_output_tokens,
        "system": config.system_prompt,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"format": {"type": "json_schema", "schema": _anthropic_schema(schema)}},
    }
    if not config.thinking:
        request["thinking"] = {"type": "disabled"}
    started = time.time()
    try:
        # Streaming keeps a large max_tokens from hitting the HTTP timeout.
        with client.messages.stream(**request) as stream:
            message = stream.get_final_message()
    except Exception as error:  # noqa: BLE001 - classify, then re-raise
        if USAGE_LIMIT_PATTERN.search(str(error)):
            raise UsageLimitError(str(error)[:400]) from error
        raise
    if message.stop_reason == "refusal":
        raise RuntimeError(f"model refused the request: {getattr(message, 'stop_details', None)}")
    if message.stop_reason == "max_tokens":
        raise RuntimeError("response hit max_tokens; raise max_output_tokens or lower chunk_size")
    text = next(block.text for block in message.content if block.type == "text")
    usage = Usage(
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        cached_input_tokens=getattr(message.usage, "cache_read_input_tokens", 0) or 0,
        duration_ms=(time.time() - started) * 1000,
    )
    return json.loads(text), usage


def call_openai(prompt: str, schema: Mapping[str, Any], config: ProviderConfig) -> tuple[Dict[str, Any], Usage]:
    """One rating request to the OpenAI Responses API."""
    from openai import OpenAI

    client = OpenAI(api_key=read_api_key("OPENAI_API_KEY"), timeout=float(config.timeout_seconds))
    started = time.time()
    try:
        response = client.responses.create(
            model=config.model,
            instructions=config.system_prompt,
            input=prompt,
            reasoning={"effort": config.reasoning_effort},
            text={"format": {"type": "json_schema", "name": "segment_ratings", "schema": dict(schema), "strict": True}},
            max_output_tokens=config.max_output_tokens,
        )
    except Exception as error:  # noqa: BLE001 - classify, then re-raise
        if USAGE_LIMIT_PATTERN.search(str(error)):
            raise UsageLimitError(str(error)[:400]) from error
        raise
    if response.status != "completed":
        raise RuntimeError(f"OpenAI response {response.status}: {getattr(response, 'incomplete_details', None)}")
    details = getattr(response.usage, "output_tokens_details", None)
    cached = getattr(getattr(response.usage, "input_tokens_details", None), "cached_tokens", 0) or 0
    usage = Usage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
        cached_input_tokens=cached,
        duration_ms=(time.time() - started) * 1000,
    )
    return json.loads(response.output_text), usage


def _run_cli(command: list, prompt: str, config: ProviderConfig, environment: Dict[str, str]) -> tuple[str, float]:
    """Run a local CLI with the prompt on stdin, in an empty working directory."""
    import subprocess
    import tempfile

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="story_ratings_") as workdir:
        try:
            completed = subprocess.run(
                command, input=prompt, text=True, encoding="utf-8", errors="replace",
                capture_output=True, check=False, timeout=config.timeout_seconds,
                env=environment, cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"{command[0]} timed out after {config.timeout_seconds}s "
                "(a blocked usage limit can look like a timeout)"
            ) from None
    output = f"{completed.stdout or ''} {completed.stderr or ''}"
    if USAGE_LIMIT_PATTERN.search(output):
        raise UsageLimitError(output.strip()[:400])
    if completed.returncode != 0:
        raise RuntimeError(f"{command[0]} exited {completed.returncode}: {output.strip()[:400]}")
    return completed.stdout, (time.time() - started) * 1000


def call_claude_cli(prompt: str, schema: Mapping[str, Any], config: ProviderConfig) -> tuple[Dict[str, Any], Usage]:
    """Rate through the local ``claude`` CLI, billed to a Claude subscription.

    The CLI refuses to nest inside a Claude Code session, so the two
    ``CLAUDECODE`` variables are dropped. Replacing the system prompt and
    disabling MCP servers and project settings keeps fixed overhead near 1K
    tokens instead of the ~44K a default session carries.

    ``thinking=False`` sets a zero thinking budget, which silences thinking on
    budget-based models (Claude Haiku 4.5) but not on models that think
    adaptively (Claude Sonnet 5). Use the ``anthropic`` backend when thinking
    must be off.
    """
    import shutil

    environment = {
        key: value for key, value in os.environ.items()
        if key not in {"CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "MAX_THINKING_TOKENS"}
    }
    if not config.thinking:
        environment["MAX_THINKING_TOKENS"] = "0"
    command = [
        shutil.which("claude") or "claude", "-p",
        "--model", config.model,
        "--system-prompt", config.system_prompt,
        "--strict-mcp-config",
        "--setting-sources", "local",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--tools", "",
        "--output-format", "json",
        "--json-schema", json.dumps(dict(schema), separators=(",", ":")),
    ]
    stdout, duration_ms = _run_cli(command, prompt, config, environment)
    wrapper = json.loads(stdout)
    if wrapper.get("is_error"):
        raise RuntimeError(f"claude reported an error: {str(wrapper.get('errors') or wrapper.get('result'))[:400]}")
    payload = wrapper.get("structured_output")
    if not isinstance(payload, dict):
        raise RuntimeError("claude returned no structured_output")
    usage = wrapper.get("usage") or {}
    return payload, Usage(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cached_input_tokens=usage.get("cache_read_input_tokens", 0) or 0,
        duration_ms=duration_ms,
    )


def call_codex_cli(prompt: str, schema: Mapping[str, Any], config: ProviderConfig) -> tuple[Dict[str, Any], Usage]:
    """Rate through the local ``codex`` CLI, billed to a ChatGPT/Codex subscription.

    Codex writes its final message to a file rather than stdout, so the schema
    and the reply travel through a scratch directory. The CLI reports no token
    counts, so usage comes back as zeros with only wall time filled in.
    """
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(prefix="story_ratings_codex_") as workdir:
        scratch = Path(workdir)
        schema_path = scratch / "schema.json"
        reply_path = scratch / "reply.json"
        schema_path.write_text(json.dumps(dict(schema)), encoding="utf-8")
        command = [
            shutil.which("codex") or "codex", "exec",
            "-m", config.model,
            "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--sandbox", "read-only", "--skip-git-repo-check",
            "-C", str(scratch),
            "--output-schema", str(schema_path),
            "--output-last-message", str(reply_path),
            "-",
        ]
        _, duration_ms = _run_cli(command, prompt, config, dict(os.environ))
        if not reply_path.exists():
            raise RuntimeError("codex wrote no reply file")
        payload = json.loads(reply_path.read_text(encoding="utf-8"))
    return payload, Usage(duration_ms=duration_ms)


def call_model(prompt: str, schema: Mapping[str, Any], config: ProviderConfig) -> tuple[Dict[str, Any], Usage]:
    """Dispatch one request to the configured backend."""
    config.validate()
    if config.backend == "anthropic":
        return call_anthropic(prompt, schema, config)
    if config.backend == "openai":
        return call_openai(prompt, schema, config)
    if config.backend == "claude-cli":
        return call_claude_cli(prompt, schema, config)
    return call_codex_cli(prompt, schema, config)
