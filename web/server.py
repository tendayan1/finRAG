"""finRAG 演示后端（Python 标准库 http.server，零新依赖）。

启动:  python -m web.server [--host 127.0.0.1] [--port 8000]
路由:
  GET  /                  首页 index.html
  GET  /api/stats         库统计（公司数 / chunk 数 / 数据表数 / 配置）
  GET  /api/examples      示例问题
  POST /api/ask           {question} -> {answer, citations, steps, elapsed_ms}
"""

import argparse
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

STATIC_DIR = Path(__file__).resolve().parent / "static"

# 示例问题（覆盖文档检索 / SQL / 路由拒答 三类）
EXAMPLES = [
    {"q": "青洲银行2022年营业收入是多少？", "tag": "文档检索"},
    {"q": "股票代码002244在2019年12月20日的收盘价是多少？", "tag": "SQL 查询"},
    {"q": "嘉实基金2019年新成立了多少只基金？", "tag": "SQL 查询"},
    {"q": "申万行业分类中建筑材料行业2019年累计涨幅超过5%的股票有哪些？", "tag": "SQL 跨表"},
    {"q": "2026 年奥运会在哪举办？", "tag": "库外拒答"},
]


def _stats() -> dict:
    """收集库统计。任何子项失败都优雅降级为 None，不让首页崩。"""
    info = {"companies": None, "chunks": None, "tables": None,
            "chat_model": None, "embedding_model": None}
    try:
        from rag_v2 import retrieve
        try:
            info["chunks"] = retrieve.load_vectorstore()._collection.count()
        except Exception:
            info["chunks"] = 0
        try:
            info["companies"] = len(retrieve.list_companies())
        except Exception:
            info["companies"] = 0
    except Exception:
        pass
    try:
        from rag_v2.sql_tools import _TABLE_SUMMARY
        names = [l.split(":")[0].lstrip("- ").strip()
                 for l in _TABLE_SUMMARY.splitlines()
                 if l.strip().startswith("- ")]
        info["tables"] = len(names)
        info["table_names"] = names
    except Exception:
        info["tables"] = 10
    try:
        from rag_v2 import config
        info["chat_model"] = config.chat_model_name()
        try:
            info["embedding_model"] = config.embedding_model_name()
        except Exception:
            info["embedding_model"] = None
    except Exception:
        pass
    return info


def _run_ask(question: str) -> dict:
    """同步跑 agent（与命令行 ask 同链路），捕获工具调用轨迹（含结果预览）。

    必须用同步 stream 而非 asyncio.run：ThreadingHTTPServer 每个请求在独立线程中
    处理，若每请求各自创建/关闭事件循环，openai(httpx/httpcore) 的连接 transport
    会被 GC 终始器带到已关闭的旧循环上，第二次请求即抛 "Event loop is closed"。
    同步客户端不触碰事件循环，可跨线程安全复用进程。
    """
    from rag_v2.agent import build_agent, RECURSION_LIMIT

    agent = build_agent()
    cfg = {"recursion_limit": RECURSION_LIMIT}
    steps = []
    answer = ""

    t0 = time.perf_counter()
    try:
        for event in agent.stream(
            {"messages": [("user", question)]}, config=cfg, stream_mode="values"
        ):
            msgs = event.get("messages", [])
            if not msgs:
                continue
            msg = msgs[-1]
            t = getattr(msg, "type", None)
            if t == "ai" and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    steps.append({
                        "kind": "call",
                        "tool": tc["name"],
                        "args": tc["args"],
                    })
            elif t == "tool":
                content = getattr(msg, "content", "") or ""
                name = getattr(msg, "name", "") or (steps[-1]["tool"] if steps else "tool")
                preview = content[:300].replace("\n", " ")
                steps.append({
                    "kind": "result",
                    "tool": name,
                    "preview": preview,
                    "len": len(content),
                })
            elif t == "ai" and getattr(msg, "content", ""):
                answer = msg.content
    except Exception as e:
        return {"answer": f"ERROR: {e}", "citations": [], "steps": steps,
                "elapsed_ms": int((time.perf_counter() - t0) * 1000)}
    elapsed = int((time.perf_counter() - t0) * 1000)

    # 后处理：偶发模型在最终答案前泄漏英文思考（如 "I now have the data..."）。
    # 仅在首个中文字符前出现英文字母时裁掉前缀；数字/符号开头（"55 只"、"4.67 元"）保留。
    import re
    if answer:
        m = re.search(r"[一-鿿]", answer)
        if m and m.start() > 0 and re.search(r"[A-Za-z]", answer[:m.start()]):
            answer = answer[m.start():]

    # 复用 agent 的引用解析逻辑
    cit_re = re.compile(r"\[来源:\s*([^\]#]+)#([^\]\s]+)\]")
    citations = [f"{m.group(1)}#{m.group(2)}" for m in cit_re.finditer(answer)]
    return {"answer": answer, "citations": citations, "steps": steps,
            "elapsed_ms": elapsed}


class Handler(BaseHTTPRequestHandler):
    server_version = "finRAG-demo/1.0"

    def _json(self, obj, code=HTTPStatus.OK):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path: Path, content_type: str):
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"Not found: {path.name}")
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("", "/", "/index.html"):
            self._static(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/api/stats":
            self._json(_stats())
            return
        if path == "/api/examples":
            self._json({"examples": EXAMPLES})
            return
        if path == "/api/health":
            self._json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/ask":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            question = (payload.get("question") or "").strip()
        except Exception as e:
            self._json({"error": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)
            return
        if not question:
            self._json({"error": "question 不能为空"}, HTTPStatus.BAD_REQUEST)
            return
        if len(question) > 2000:
            self._json({"error": "question 过长（>2000 字符）"}, HTTPStatus.BAD_REQUEST)
            return
        result = _run_ask(question)
        self._json(result)

    def log_message(self, fmt, *args):
        # 简化日志：仅打方法 + 路径 + 状态
        try:
            line = fmt % args
        except Exception:
            line = " ".join(str(a) for a in args)
        # 过滤静态资源噪声
        if "/static/" in line or " 304 " in line:
            return
        print(f"[{self.address_string()}] {line}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="finRAG 演示服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"finRAG 演示服务已启动: {url}")
    print("  GET  /                 首页")
    print("  GET  /api/stats        库统计")
    print("  GET  /api/examples     示例问题")
    print("  POST /api/ask          提问")
    print("Ctrl+C 退出")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
