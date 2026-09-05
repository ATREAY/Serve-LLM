"""OpenAI-compatible request/response schemas (chat + completions, streaming included).

Subset of the OpenAI API surface: enough for chat/completion clients (openai-python,
LangChain, curl) to talk to ServeLLM without modification.
"""

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


# ---- Chat ----

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = None
    n: int = 1
    stream: bool = False
    stop: list[str] | str | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None
    # ServeLLM extension, not part of the OpenAI API: lower value = served
    # first (matches vLLM's own scheduler convention directly). Only has any
    # effect on a model configured with scheduling_policy: priority in
    # models.yaml — see backend/router/registry.py.
    priority: int = 0


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _id("chatcmpl"))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionChunkDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: _id("chatcmpl"))
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]


# ---- Completions (legacy /v1/completions) ----

class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = 16
    n: int = 1
    stream: bool = False
    stop: list[str] | str | None = None
    priority: int = 0  # ServeLLM extension — see ChatCompletionRequest.priority


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _id("cmpl"))
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


# ---- Models ----

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "servellm"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


# ---- Admin: Phase 4 dynamic adapters ----

class AdapterRegisterRequest(BaseModel):
    base_model: str
    name: str
    hf_repo: str
    version: str = "v1"


class AdapterInfo(BaseModel):
    base_model: str
    name: str
    hf_repo: str
    version: str
    status: str
    hits: int
    last_used_at: str | None = None
    created_at: str


# ---- Admin: Phase 13 security ----

class TokenRequest(BaseModel):
    api_key: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class RequestLogEntry(BaseModel):
    request_id: str
    model: str
    adapter: str | None
    endpoint: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    ttft_seconds: float | None
    status: str
    created_at: str
