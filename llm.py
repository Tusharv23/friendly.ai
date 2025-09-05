"""Shared LLM access helpers (Bedrock Claude, optionally others)."""
from __future__ import annotations

import json
from typing import List, Dict, Optional
import boto3

CLAUDE_DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"

def _bedrock_client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")

def bedrock_claude_chat(
    messages: List[Dict[str, str]],
    model: str = CLAUDE_DEFAULT_MODEL,
    max_tokens: int = 400,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    """Send chat-style messages to Claude via Bedrock Messages API.

    messages: list of {role: 'user'|'assistant', content: str}
    Returns aggregated text content.
    """
    # Convert simple messages list into API message format expected.
    api_messages = []
    for m in messages:
        api_messages.append({
            "role": m["role"],
            "content": [{"type": "text", "text": m["content"]}]
        })

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": api_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    br = _bedrock_client()
    resp = br.invoke_model(
        modelId=model,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    data = json.loads(resp["body"].read())
    # Claude returns list of content blocks.
    text_parts = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    out = "".join(text_parts).strip()
    if out:
        return out
    return data.get("output_text", "")
