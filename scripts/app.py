"""FastAPI inference server with a simple chat page.

Usage:
    python scripts/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from scripts.evaluate import load_config, load_generator  # noqa: E402


app = FastAPI(title="Finance SFT Qwen")
config = load_config(str(ROOT / "configs" / "eval.yaml"))

model = None
tokenizer = None


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


def get_model_and_tokenizer():
    global model, tokenizer
    if model is None:
        model, tokenizer = load_generator(
            config.model_name_or_path,
            "outputs/checkpoints/final",
        )
    return model, tokenizer


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE_HTML


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is empty")

    loaded_model, loaded_tokenizer = get_model_and_tokenizer()
    messages = [{"role": "user", "content": question}]
    prompt = loaded_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = loaded_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(loaded_model.device)
    with torch.no_grad():
        output_ids = loaded_model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            pad_token_id=loaded_tokenizer.pad_token_id,
            eos_token_id=loaded_tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][prompt_len:]
    answer = loaded_tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()
    return ChatResponse(answer=answer)


PAGE_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Finance SFT Qwen</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: #f4f6f8;
      color: #1f2933;
      font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
      display: flex;
      justify-content: center;
    }
    .app {
      width: min(760px, 100%);
      display: flex;
      flex-direction: column;
      padding: 28px 18px;
    }
    header { margin-bottom: 18px; }
    h1 { font-size: 20px; margin: 0 0 6px; }
    .sub { color: #64748b; font-size: 13px; }
    main {
      flex: 1;
      min-height: 420px;
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 16px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .msg {
      max-width: 88%;
      padding: 10px 12px;
      border-radius: 8px;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.55;
    }
    .user {
      align-self: flex-end;
      background: #2563eb;
      color: #ffffff;
    }
    .assistant {
      align-self: flex-start;
      background: #f1f5f9;
    }
    footer { display: flex; gap: 8px; margin-top: 14px; }
    input {
      flex: 1;
      padding: 12px 14px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font-size: 15px;
    }
    button {
      padding: 12px 18px;
      border: 0;
      border-radius: 8px;
      background: #2563eb;
      color: #ffffff;
      font-size: 15px;
      cursor: pointer;
    }
    button:disabled { opacity: 0.6; cursor: default; }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>Finance SFT Qwen</h1>
      <div class="sub">Qwen3-4B-Instruct-2507 + QLoRA SFT</div>
    </header>
    <main id="chat"></main>
    <footer>
      <input id="question" placeholder="输入你的金融问题" autocomplete="off">
      <button id="send">发送</button>
    </footer>
  </div>
  <script>
    const chat = document.getElementById("chat");
    const input = document.getElementById("question");
    const send = document.getElementById("send");

    function append(text, cls) {
      const el = document.createElement("div");
      el.className = "msg " + cls;
      el.textContent = text;
      chat.appendChild(el);
      chat.scrollTop = chat.scrollHeight;
      return el;
    }

    async function ask() {
      const question = input.value.trim();
      if (!question) return;
      input.value = "";
      append(question, "user");
      send.disabled = true;
      const waiting = append("正在生成...", "assistant");
      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question})
        });
        const data = await response.json();
        waiting.textContent = data.answer || "没有生成答案";
      } catch (error) {
        waiting.textContent = "请求失败，请稍后重试";
      } finally {
        send.disabled = false;
        input.focus();
      }
    }

    send.addEventListener("click", ask);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") ask();
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
