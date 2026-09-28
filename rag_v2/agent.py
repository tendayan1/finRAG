"""LangGraph ReAct agent 组装与问答入口。

agent 自主决定：是否检索 -> 用什么 query -> 是否换 query 再检索 -> 何时作答。
"""

import re

from langgraph.prebuilt import create_react_agent

from .config import get_chat_model
from .prompts import SYSTEM_PROMPT
from .tools import ALL_TOOLS

RECURSION_LIMIT = 40  # 复杂分析题（跨表 JOIN、全年聚合+多轮 SQL 修正）需要更多迭代步数

_CITATION_RE = re.compile(r"\[来源:\s*([^\]#]+)#([^\]\s]+)\]")


def build_agent():
    return create_react_agent(get_chat_model(), tools=ALL_TOOLS, prompt=SYSTEM_PROMPT)


def ask(question: str, verbose: bool = False) -> dict:
    """提问，返回 {answer, citations, steps}。verbose 时打印 tool call 轨迹。"""
    agent = build_agent()
    config = {"recursion_limit": RECURSION_LIMIT}
    steps = []
    answer = ""

    for event in agent.stream(
        {"messages": [("user", question)]}, config=config, stream_mode="values"
    ):
        answer = _handle_event(event, steps, verbose) or answer

    return _result(answer, steps)


async def aask(question: str, verbose: bool = False) -> dict:
    """ask 的异步版本，供批量并发调用。"""
    agent = build_agent()
    config = {"recursion_limit": RECURSION_LIMIT}
    steps = []
    answer = ""

    async for event in agent.astream(
        {"messages": [("user", question)]}, config=config, stream_mode="values"
    ):
        answer = _handle_event(event, steps, verbose) or answer

    return _result(answer, steps)


def _handle_event(event: dict, steps: list, verbose: bool) -> str:
    """处理一条 stream 事件，返回最新的 answer 文本（无则空串）。"""
    msg = event["messages"][-1]
    if msg.type == "ai" and msg.tool_calls:
        for tc in msg.tool_calls:
            steps.append({"tool": tc["name"], "args": tc["args"]})
            if verbose:
                print(f"[tool_call] {tc['name']}({tc['args']})")
    elif msg.type == "tool" and verbose:
        preview = msg.content[:120].replace("\n", " ")
        print(f"[tool_result] {msg.name}: {preview}...")
    elif msg.type == "ai" and msg.content:
        return msg.content
    return ""


def _result(answer: str, steps: list) -> dict:
    return {
        "answer": answer,
        "citations": [f"{m.group(1)}#{m.group(2)}" for m in _CITATION_RE.finditer(answer)],
        "steps": steps,
    }
