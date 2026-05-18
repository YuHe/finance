"""LLM 提供商注册 + OpenAI 协议统一调用"""

from dataclasses import dataclass
from typing import Optional
import httpx


# 预制提供商
PROVIDERS = {
    "dashscope": {
        "name": "通义千问 (DashScope)",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": [
            {"id": "qwen-plus", "name": "Qwen Plus", "supports_search": True},
            {"id": "qwen-max", "name": "Qwen Max", "supports_search": True},
            {"id": "qwen-turbo", "name": "Qwen Turbo", "supports_search": True},
        ],
        "supports_web_search": True,
    },
    "deepseek": {
        "name": "DeepSeek",
        "api_base": "https://api.deepseek.com/v1",
        "models": [
            {"id": "deepseek-chat", "name": "DeepSeek Chat", "supports_search": False},
            {"id": "deepseek-reasoner", "name": "DeepSeek Reasoner", "supports_search": False},
        ],
        "supports_web_search": False,
    },
}


@dataclass
class LLMRequest:
    messages: list[dict]
    model: str
    api_base: str
    api_key: str
    web_search: bool = False
    temperature: float = 0.3
    max_tokens: int = 2000


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict
    success: bool
    error: Optional[str] = None


def _build_body(req: LLMRequest) -> dict:
    body = {
        "model": req.model,
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    if req.web_search:
        # DashScope OpenAI-compatible 模式: enable_search
        if "dashscope" in req.api_base:
            body["enable_search"] = True
        else:
            # 通用 OpenAI 协议的工具调用方式
            body["tools"] = [{"type": "web_search", "web_search": {"enable": True}}]
    return body


async def call_llm(req: LLMRequest) -> LLMResponse:
    """调用 OpenAI 协议兼容的 LLM API"""
    url = f"{req.api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {req.api_key}",
        "Content-Type": "application/json",
    }
    body = _build_body(req)

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return LLMResponse(content=content, model=req.model, usage=usage, success=True)
        except httpx.HTTPStatusError as e:
            return LLMResponse(
                content="", model=req.model, usage={}, success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            )
        except Exception as e:
            return LLMResponse(
                content="", model=req.model, usage={}, success=False,
                error=str(e)
            )
