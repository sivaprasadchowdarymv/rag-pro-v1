"""
Offline fake of the Groq (OpenAI-compatible) and Ollama APIs, for tests only.

Scripted behaviour is chosen from the question text, and protocol rules are
validated strictly (every tool call answered with its id; Ollama tool replies
carry tool_name). Control: GET /_ctl?fail=groq|ollama|none, GET /_log.
"""
from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

STATE = {"fail": "none", "requests": [], "errors": []}
GROQ_KEY, OLLAMA_KEY = "gsk_test", "ollama_test"
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
OLLAMA_MODELS = ["gpt-oss:120b", "gpt-oss:20b", "gemma4:31b"]
LABEL = re.compile(r"\[REF-\d+: [A-Z]+ p\d+\]")


def scripted(messages, tools_on):
    """Return (content, tool_calls[list of (name, args)]) like a cooperative model."""
    text = "\n".join(m.get("content") or "" if isinstance(m.get("content"), str) else "" for m in messages)
    if "strict verifier" in text:
        return json.dumps({"grounded": True, "unsupported_claims": [], "notes": "ok"}), []
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    q = (re.search(r"Question:\s*(.*)", user) or re.search(r"(.*)", user)).group(1).lower()
    labels = LABEL.findall(text)
    n_tool = sum(1 for m in messages if m["role"] == "tool")
    if ("dissipation" in q or "calculate" in q) and tools_on and n_tool == 0:
        return "", [("search_datasheet", {"query": "dropout voltage", "source": "equation"}),
                    ("calculate", {"expression": "(12-5)*1.5"})]
    if "loop" in q and tools_on:
        return "", [("read_page", {"page": 1})]
    if "nothing" in q or not labels:
        return "NOT FOUND IN DOCUMENT", []
    a, b = labels[0], labels[min(1, len(labels) - 1)]
    if "unsupported number" in q and "VERIFICATION FEEDBACK" not in text:
        return f"**Answer:** The output current is 9.87 A {a}.", []
    if "invent" in q:
        return "**Answer:** It is 3.3 V [REF-99: TEXT p1].", []
    if n_tool:  # answer the calculation question from the tool observations
        return (f"**Answer:** The power dissipation is 10.5 W {a}.\n\n**Equations:**\n\n"
                "$$P_D = (V_{IN} - V_{OUT}) \\times I_{OUT}$$\n\n"
                f"Using the calculator: $P_D = (12-5) \\times 1.5 = 10.5$ W {a}"), []
    return (f"**Answer:** The peak output current is 2.2 A {a}.\n\n"
            f"**Explanation:** This is the maximum current the regulator delivers {b}.\n\n"
            "**Equations:**\n\n$$P_D = (V_{IN} - V_{OUT}) \\times I_{OUT}$$\n\nwhere $P_D$ is the power dissipation "
            + a), []


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj, headers=None):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _auth(self, key, optional=False):
        got = self.headers.get("Authorization")
        if optional and got is None:
            return True
        if got != f"Bearer {key}":
            self._send(401, {"error": {"message": "invalid api key", "code": "invalid_api_key"}})
            return False
        return True

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/_ctl":
            STATE["fail"] = parse_qs(url.query).get("fail", ["none"])[0]
            return self._send(200, {"fail": STATE["fail"]})
        if url.path == "/_log":
            return self._send(200, STATE)
        if url.path == "/_reset":
            STATE["requests"].clear(); STATE["errors"].clear(); STATE["fail"] = "none"
            return self._send(200, {"ok": True})
        if url.path == "/openai/v1/models":
            if not self._auth(GROQ_KEY):
                return
            return self._send(200, {"object": "list", "data": [
                {"id": i, "object": "model", "created": 0, "owned_by": "x"} for i in GROQ_MODELS + ["whisper-large-v3"]]})
        if url.path == "/api/tags":
            if not self._auth(OLLAMA_KEY, optional=True):
                return
            return self._send(200, {"models": [{"name": m, "model": m, "modified_at": "2026-01-01T00:00:00Z",
                                                "size": 1, "digest": "x"} for m in OLLAMA_MODELS]})
        self._send(404, {"error": "nf"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/openai/v1/chat/completions":
            return self._groq(body)
        if self.path == "/api/chat":
            return self._ollama(body)
        self._send(404, {"error": "nf"})

    # ---------------------------------------------------------------- Groq
    def _groq(self, body):
        if not self._auth(GROQ_KEY):
            return
        STATE["requests"].append(("groq", body["model"]))
        if STATE["fail"] == "groq":
            return self._send(429, {"error": {"message": "rate limit", "code": "rate_limit_exceeded"}}, {"retry-after": "0"})
        if body["model"] not in GROQ_MODELS:
            return self._send(404, {"error": {"message": "model not found", "code": "model_not_found"}})
        pending = set()
        for m in body["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                pending |= {t["id"] for t in m["tool_calls"]}
            elif m["role"] == "tool":
                if m["tool_call_id"] not in pending:
                    STATE["errors"].append("groq: tool reply without call")
                pending.discard(m["tool_call_id"])
            elif pending:
                STATE["errors"].append("groq: message before tool calls answered")
        tools_on = bool(body.get("tools")) and body.get("tool_choice") != "none"
        content, calls = scripted(body["messages"], tools_on)
        msg = {"role": "assistant", "content": content or None}
        if calls:
            msg["tool_calls"] = [{"id": f"call_{i}_{len(STATE['requests'])}", "type": "function",
                                  "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
        chars = len(json.dumps(body["messages"]))
        self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                         "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                         "usage": {"prompt_tokens": chars // 4, "completion_tokens": 100, "total_tokens": chars // 4 + 100}})

    # -------------------------------------------------------------- Ollama
    def _ollama(self, body):
        if not self._auth(OLLAMA_KEY, optional=True):
            return
        STATE["requests"].append(("ollama", body["model"]))
        if STATE["fail"] == "ollama":
            return self._send(429, {"error": "usage limit reached"})
        if body["model"] not in OLLAMA_MODELS:
            return self._send(404, {"error": f"model '{body['model']}' not found"})
        for m in body["messages"]:
            if m["role"] == "tool" and not m.get("tool_name"):
                STATE["errors"].append("ollama: tool message without tool_name")
        if "tool_choice" in body:
            STATE["errors"].append("ollama: tool_choice is not an Ollama parameter")
        content, calls = scripted(body["messages"], bool(body.get("tools")))
        msg = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = [{"function": {"name": n, "arguments": a}} for n, a in calls]
        self._send(200, {"model": body["model"], "created_at": "2026-01-01T00:00:00Z", "message": msg,
                         "done": True, "prompt_eval_count": len(json.dumps(body["messages"])) // 4, "eval_count": 100})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
