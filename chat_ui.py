from __future__ import annotations

import argparse
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import gradio as gr

from src.ascii_llm.runtime import generate_ascii, load_model_and_tokenizer, read_config

LOGGER = logging.getLogger("ascii_llm.chat_ui")

DEFAULT_QUICK_PROMPTS = [
    "draw a dragon made of flames",
    "draw a retro arcade spaceship",
    "draw a detailed cat face",
    "draw a skull in gothic style",
    "draw a mountain landscape at night",
    "draw a stylized text logo: ASCII",
]

CUSTOM_CSS = """
:root {
  --bg-a: #fff8ea;
  --bg-b: #d7f1ec;
  --bg-c: #f3eee6;
  --ink: #10243d;
  --ink-soft: #4f637f;
  --card: #ffffff;
  --card-edge: #d9e2ef;
}

.gradio-container {
  font-family: "Sora", "Segoe UI", sans-serif !important;
  background:
    radial-gradient(circle at 12% 10%, rgba(255, 167, 76, 0.25), transparent 28%),
    radial-gradient(circle at 82% 22%, rgba(45, 169, 143, 0.25), transparent 32%),
    linear-gradient(135deg, var(--bg-a), var(--bg-b) 45%, var(--bg-c));
  min-height: 100vh;
}

#shell {
  max-width: 1500px;
  margin: 0 auto;
  padding: 20px;
}

#main-pane, #side-pane {
  border: 1px solid var(--card-edge);
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(6px);
}

#main-pane {
  min-height: calc(100vh - 48px);
  padding: 18px;
}

#side-pane {
  min-height: calc(100vh - 48px);
  padding: 14px;
}

#brand {
  margin-bottom: 8px;
}

#brand h1 {
  margin: 0;
  color: var(--ink);
  font-size: 1.5rem;
}

#brand p {
  margin: 2px 0 0;
  color: var(--ink-soft);
  font-size: 0.9rem;
}

#status-box {
  border: 1px solid var(--card-edge);
  border-radius: 12px;
  background: #f8fbff;
  color: var(--ink-soft);
  padding: 8px 10px;
  margin-bottom: 10px;
}

#chatbot {
  border: 1px solid var(--card-edge);
  border-radius: 14px;
  overflow: hidden;
}

#chatbot .message {
  font-family: "JetBrains Mono", "Consolas", monospace !important;
  line-height: 1.25;
}

#chatbot .user {
  background: rgba(255, 122, 61, 0.12);
  border: 1px solid rgba(255, 122, 61, 0.32);
}

#chatbot .bot {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(16, 36, 61, 0.14);
}

#prompt-box textarea {
  font-family: "JetBrains Mono", "Consolas", monospace !important;
  border-radius: 12px !important;
}

#chat-tabs .wrap label {
  border: 1px solid var(--card-edge);
  border-radius: 10px;
  margin-bottom: 6px;
  background: #f9fcff;
}

#chat-tabs .wrap label.selected {
  border-color: #2b9f8e;
  background: rgba(43, 159, 142, 0.14);
}

#side-pane button.primary {
  background: #10243d !important;
  color: #f7fafc !important;
}

#side-pane button.secondary {
  background: #ffffff !important;
  color: #10243d !important;
}

@media (max-width: 1024px) {
  #main-pane, #side-pane {
    min-height: auto;
  }
  #side-pane {
    margin-top: 12px;
  }
}
"""


@dataclass
class AppSettings:
    config_path: str
    adapter_path: str | None
    load_in_4bit: bool


@dataclass
class RuntimeCache:
    settings: AppSettings
    config: dict[str, Any] | None = None
    model: Any | None = None
    tokenizer: Any | None = None
    mode: str = "uninitialized"
    lock: threading.Lock = field(default_factory=threading.Lock)
    infer_lock: threading.Lock = field(default_factory=threading.Lock)

    def ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None and self.config is not None:
            return
        with self.lock:
            if self.model is not None and self.tokenizer is not None and self.config is not None:
                return
            self.config = read_config(self.settings.config_path)
            self.model, self.tokenizer, self.mode = load_model_and_tokenizer(
                config=self.config,
                adapter_path=self.settings.adapter_path,
                load_in_4bit=self.settings.load_in_4bit,
            )


SETTINGS: AppSettings | None = None
RUNTIME: RuntimeCache | None = None


def _make_chat_title(prompt: str, index: int) -> str:
    cleaned = " ".join(prompt.strip().split())
    if not cleaned:
        return f"Chat {index}"
    return cleaned[:28] + ("..." if len(cleaned) > 28 else "")


def _new_chat_state(counter: int) -> tuple[str, dict[str, Any]]:
    chat_id = f"chat-{uuid.uuid4().hex[:8]}"
    return chat_id, {"title": f"Chat {counter}", "messages": []}


def _init_state() -> dict[str, Any]:
    chat_id, chat = _new_chat_state(counter=1)
    return {
        "counter": 1,
        "active": chat_id,
        "order": [chat_id],
        "chats": {chat_id: chat},
    }


def _choices(state: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for chat_id in state["order"]:
        out.append((state["chats"][chat_id]["title"], chat_id))
    return out


def _active_messages(state: dict[str, Any]) -> list[dict[str, str]]:
    chat_id = state["active"]
    return state["chats"][chat_id]["messages"]


def _status_line(prefix: str = "Ready") -> str:
    assert SETTINGS is not None
    adapter = SETTINGS.adapter_path or "auto"
    bit = "4bit" if SETTINGS.load_in_4bit else "full precision"
    return f"{prefix} | config={SETTINGS.config_path} | adapter={adapter} | {bit}"


def _build_prompt_from_messages(messages: list[dict[str, str]]) -> str:
    clean = [m for m in messages if m["role"] in {"user", "assistant"} and m["content"].strip()]
    if not clean:
        return ""
    if len(clean) == 1 and clean[0]["role"] == "user":
        return clean[0]["content"].strip()
    parts = ["Conversation transcript:"]
    for msg in clean[-12:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        text = msg["content"].strip()
        if len(text) > 2400:
            text = text[:2400].rstrip() + "\n...[truncated]"
        parts.append(f"{role}: {text}")
    parts.append("Write only the next assistant response.")
    return "\n\n".join(parts)


def _update_chat_selector(state: dict[str, Any]) -> dict[str, Any]:
    return gr.update(choices=_choices(state), value=state["active"])


def on_new_chat(state: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], str]:
    state["counter"] += 1
    chat_id, chat = _new_chat_state(counter=state["counter"])
    state["active"] = chat_id
    state["order"].insert(0, chat_id)
    state["chats"][chat_id] = chat
    return [], state, _update_chat_selector(state), _status_line("New chat created")


def on_delete_chat(state: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], str]:
    if len(state["order"]) <= 1:
        state["chats"][state["active"]]["messages"] = []
        return [], state, _update_chat_selector(state), _status_line("Cannot delete last chat, cleared instead")

    active = state["active"]
    state["order"] = [cid for cid in state["order"] if cid != active]
    state["chats"].pop(active, None)
    state["active"] = state["order"][0]
    return _active_messages(state), state, _update_chat_selector(state), _status_line("Chat deleted")


def on_switch_chat(selected_chat: str, state: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any], str]:
    if selected_chat and selected_chat in state["chats"]:
        state["active"] = selected_chat
    return _active_messages(state), state, _status_line("Switched chat")


def on_clear_chat(state: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any], str]:
    state["chats"][state["active"]]["messages"] = []
    return [], state, _status_line("Active chat cleared")


def on_quick_prompt(selected: str) -> str:
    return selected or ""


def on_send(
    user_text: str,
    state: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], str, str]:
    text = (user_text or "").strip()
    if not text:
        return _active_messages(state), state, _update_chat_selector(state), _status_line("Empty prompt"), ""

    active = state["active"]
    chat = state["chats"][active]
    chat["messages"].append({"role": "user", "content": text})

    if chat["title"].startswith("Chat "):
        idx = max(1, state["counter"])
        chat["title"] = _make_chat_title(text, idx)

    try:
        assert RUNTIME is not None
        started = time.perf_counter()
        RUNTIME.ensure_loaded()
        assert RUNTIME.config is not None
        assert RUNTIME.model is not None
        assert RUNTIME.tokenizer is not None

        generation_config = dict(RUNTIME.config.get("generation", {}))
        generation_config["max_new_tokens"] = int(max_new_tokens)
        generation_config["temperature"] = float(temperature)
        generation_config["top_p"] = float(top_p)
        generation_config["repetition_penalty"] = float(repetition_penalty)

        prompt = _build_prompt_from_messages(chat["messages"])
        if not prompt:
            raise RuntimeError("Prompt is empty after formatting.")

        with RUNTIME.infer_lock:
            output = generate_ascii(
                model=RUNTIME.model,
                tokenizer=RUNTIME.tokenizer,
                user_prompt=prompt,
                system_prompt=RUNTIME.config.get("system_prompt", ""),
                generation_config=generation_config,
                max_new_tokens_override=max_new_tokens,
            )

        chat["messages"].append({"role": "assistant", "content": output})
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        status = _status_line(f"Generated in {elapsed_ms:.0f} ms | mode={RUNTIME.mode}")
    except Exception as exc:
        error = f"[ERROR] {exc}"
        chat["messages"].append({"role": "assistant", "content": error})
        status = _status_line(f"Generation failed: {exc}")
        LOGGER.exception("Generation failed: %s", exc)

    return _active_messages(state), state, _update_chat_selector(state), status, ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="ASCII Chat Studio") as demo:
        state = gr.State(value=_init_state())

        with gr.Row(elem_id="shell"):
            with gr.Column(scale=4, elem_id="main-pane"):
                gr.HTML(
                    '<div id="brand"><h1>ASCII Chat Studio</h1><p>Large chat area + right chat tabs</p></div>'
                )
                status = gr.Markdown(_status_line("Ready"), elem_id="status-box")
                chatbot = gr.Chatbot(type="messages", elem_id="chatbot", height=620, show_copy_button=True)
                prompt = gr.Textbox(
                    label="",
                    placeholder="Ask anything about ASCII art...",
                    lines=4,
                    elem_id="prompt-box",
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    send_btn = gr.Button("Send", variant="primary")

            with gr.Column(scale=2, elem_id="side-pane"):
                gr.Markdown("## Chats")
                chat_selector = gr.Radio(
                    choices=[],
                    value=None,
                    interactive=True,
                    label="Tabs",
                    elem_id="chat-tabs",
                )
                with gr.Row():
                    new_chat_btn = gr.Button("New chat", variant="primary")
                    delete_chat_btn = gr.Button("Delete chat", variant="secondary")

                gr.Markdown("## Generation")
                max_new_tokens = gr.Slider(32, 2048, value=320, step=1, label="Max new tokens")
                temperature = gr.Slider(0.10, 1.80, value=0.80, step=0.01, label="Temperature")
                top_p = gr.Slider(0.10, 1.00, value=0.92, step=0.01, label="Top-p")
                repetition_penalty = gr.Slider(0.90, 1.80, value=1.05, step=0.01, label="Repetition penalty")

                gr.Markdown("## Quick prompts")
                quick_prompt = gr.Dropdown(
                    choices=DEFAULT_QUICK_PROMPTS,
                    value=DEFAULT_QUICK_PROMPTS[0],
                    label="Pick a prompt",
                )
                use_prompt_btn = gr.Button("Use in prompt box")

        def _load_initial(
            state_value: dict[str, Any],
        ) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], str]:
            return _active_messages(state_value), state_value, _update_chat_selector(state_value), _status_line("Ready")

        demo.load(
            fn=_load_initial,
            inputs=[state],
            outputs=[chatbot, state, chat_selector, status],
        )

        send_btn.click(
            fn=on_send,
            inputs=[prompt, state, max_new_tokens, temperature, top_p, repetition_penalty],
            outputs=[chatbot, state, chat_selector, status, prompt],
        )
        prompt.submit(
            fn=on_send,
            inputs=[prompt, state, max_new_tokens, temperature, top_p, repetition_penalty],
            outputs=[chatbot, state, chat_selector, status, prompt],
        )

        clear_btn.click(
            fn=on_clear_chat,
            inputs=[state],
            outputs=[chatbot, state, status],
        )
        new_chat_btn.click(
            fn=on_new_chat,
            inputs=[state],
            outputs=[chatbot, state, chat_selector, status],
        )
        delete_chat_btn.click(
            fn=on_delete_chat,
            inputs=[state],
            outputs=[chatbot, state, chat_selector, status],
        )
        chat_selector.change(
            fn=on_switch_chat,
            inputs=[chat_selector, state],
            outputs=[chatbot, state, status],
        )
        use_prompt_btn.click(fn=on_quick_prompt, inputs=[quick_prompt], outputs=[prompt])

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ASCII chat UI (single script, no separate backend).")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--adapter-path", type=str, default=None, help="Optional adapter path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for Gradio server")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--inbrowser", action="store_true", help="Open browser automatically")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized loading")
    return parser.parse_args()


def main() -> None:
    global SETTINGS
    global RUNTIME

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    SETTINGS = AppSettings(
        config_path=args.config,
        adapter_path=args.adapter_path,
        load_in_4bit=not args.no_4bit,
    )
    RUNTIME = RuntimeCache(settings=SETTINGS)

    demo = build_ui()
    demo.queue(max_size=16).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_api=False,
    )


if __name__ == "__main__":
    main()
