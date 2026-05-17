import json
import os

import requests
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

app = Flask(__name__)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"


def build_payload(messages, stream, think):
    return {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "thinking": {"type": "enabled" if think else "disabled"},
    }


def build_headers():
    return {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "未检测到 DEEPSEEK_API_KEY 环境变量"}), 500

    data = request.get_json(force=True)
    messages = data.get("messages", [])
    think = bool(data.get("think", False))

    resp = requests.post(
        DEEPSEEK_BASE_URL,
        headers=build_headers(),
        json=build_payload(messages, stream=False, think=think),
        timeout=300,
    )
    if resp.status_code != 200:
        return jsonify({"error": f"DeepSeek API 错误: {resp.status_code} {resp.text}"}), 500

    result = resp.json()
    msg = result["choices"][0]["message"]
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "") or ""
    usage = result.get("usage", {})
    return jsonify({"content": content, "reasoning": reasoning, "usage": usage})


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "未检测到 DEEPSEEK_API_KEY 环境变量"}), 500

    data = request.get_json(force=True)
    messages = data.get("messages", [])
    think = bool(data.get("think", False))

    def generate():
        with requests.post(
                DEEPSEEK_BASE_URL,
                headers=build_headers(),
                json=build_payload(messages, stream=True, think=think),
                stream=True,
                timeout=300,
        ) as r:
            if r.status_code != 200:
                err = {"error": f"DeepSeek API 错误: {r.status_code} {r.text}"}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                return

            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if not raw_line.startswith("data:"):
                    continue
                payload = raw_line[len("data:"):].strip()
                if payload == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                try:
                    obj = json.loads(payload)
                    delta = obj["choices"][0].get("delta", {})
                    reasoning_piece = delta.get("reasoning_content") or ""
                    content_piece = delta.get("content") or ""
                    if reasoning_piece:
                        out = {"reasoning": reasoning_piece}
                        yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
                    if content_piece:
                        out = {"content": content_piece}
                        yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
                except Exception as e:
                    err = {"error": f"解析流数据失败: {e}"}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
