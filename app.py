from __future__ import annotations

import os
from typing import Any

import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # This loads variables from a .env file

from appsupport import build_multimodal_messages


OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL')
MODEL_CHOICES = [
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-haiku",
]
APP_TITLE = "Let Me Solve!"


def stream_advanced_chat(
    message: dict[str, Any],
    history: list[dict[str, Any]],
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
):
    """Stream a configurable multimodal response into Gradio."""
    api_key = (api_key or "").strip()
    if not api_key:
        yield "Add your OpenRouter API key first."
        return

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    response = client.chat.completions.create(
        model=(model or DEFAULT_MODEL).strip(),
        messages=build_multimodal_messages(history, message),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        extra_body={"provider": {"data_collection": "deny"}}
    )

    answer = ""
    for chunk in response:
        # Check if choices exists in this specific chunk
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                answer += content
                yield answer

def build_demo() -> gr.ChatInterface:
    """Create the advanced Gradio app with configurable controls."""
    api_key_input = gr.Textbox(label="OpenRouter API Key", type="password")
    model_input = gr.Dropdown(MODEL_CHOICES, label="Choose an option")
    temperature_input = gr.Slider(minimum=0, maximum=1.5, label="Temperature")    
    max_tokens_input = gr.Slider(minimum=64, maximum=2048, label="Max Tokens")

    return gr.ChatInterface(fn=stream_advanced_chat
    ,multimodal=True
    , title=APP_TITLE
    , textbox = gr.MultimodalTextbox()
    , additional_inputs=[api_key_input, model_input, temperature_input, max_tokens_input])


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
