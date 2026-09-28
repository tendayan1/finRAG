"""LLM 裁判抽样评估：从 results 抽样，裁判 agent 独立查证后与原答案比对。

用法:
  python -m rag_v2.evaluate --sample 30 --out eval_report.json
  python -m rag_v2.evaluate --sample 5 --verbose  # 小验证
"""

import argparse
import asyncio
import json
import random
import re
import pathlib

from langgraph.prebuilt import create_react_agent

from .config import get_chat_model
from .tools import ALL_TOOLS

RECURSION_LIMIT = 40

JUDGE_PROMPT = """你是一个答案验证裁判。下面给你一个金融问题和待评答案，请独立使用工具查证，然后判断待评答案是否正确。

问题: {question}
待评答案: {answer}

工作流程:
1. 独立使用工具（SQL 查询或文档检索）查证问题所需的数值/事实。
2. 将你查到的正确结果与待评答案对比（注意数字精度、日期格式、保留位数要求）。
3. 给出判定。

判定标准:
- correct: 你查到的结果与待评答案一致（允许取整/保留位数差异）
- partially_correct: 方向正确但有数值偏差或遗漏部分信息
- incorrect: 你查到的结果与待评答案矛盾
- unverifiable: 待评答案是拒答，且你也查不到（视为合理拒答→correct；你查得到→incorrect）

最终输出一行 JSON（不要输出其他内容）:
{{"verdict": "correct|partially_correct|incorrect", "ground_truth": "你查到的正确值", "reason": "简短说明（中文）"}}"""


def load_results(progress_path: str) -> list[dict]:
    p = pathlib.Path(progress_path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def stratified_sample(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    """分层抽样：有引用的（文档题）和无引用的（SQL 题）各取一半。"""
    rng = random.Random(seed)
    cited = [r for r in records if r.get("citations")]
    uncited = [r for r in records if not r.get("citations")]
    half = n // 2
    return rng.sample(cited, min(half, len(cited))) + rng.sample(uncited, min(n - half, len(uncited)))


async def judge_one(rec: dict, sem: asyncio.Semaphore, verbose: bool) -> dict:
    """裁判单题：独立 agent 查证后输出 verdict。"""
    async with sem:
        prompt = JUDGE_PROMPT.format(question=rec["question"], answer=rec["answer"])
        agent = create_react_agent(get_chat_model(), tools=ALL_TOOLS, prompt=prompt)
        config = {"recursion_limit": RECURSION_LIMIT}
        raw = ""
        try:
            async for event in agent.astream(
                {"messages": [("user", "请验证上述答案。")]}, config=config, stream_mode="values"
            ):
                msg = event["messages"][-1]
                if msg.type == "ai" and msg.content:
                    raw = msg.content
        except Exception as e:
            raw = f'{{"verdict": "unverifiable", "ground_truth": "", "reason": "裁判异常: {e}"}}'

        # 解析 JSON（agent 可能包裹在 markdown 中）
        m = re.search(r'\{[^{}]*"verdict"[^{}]*\}', raw, re.DOTALL)
        verdict_info = {"verdict": "unverifiable", "ground_truth": "", "reason": raw[:200]}
        if m:
            try:
                verdict_info = json.loads(m.group(0))
            except json.JSONDecodeError:
                verdict_info["reason"] = raw[:200]

        if verbose:
            print(f"[{rec['key']}] {verdict_info['verdict']}: {verdict_info.get('ground_truth','')} "
                  f"vs 答案={rec['answer'][:60].replace(chr(10),' ')}")

        return {
            "id": rec["key"],
            "question": rec["question"][:80],
            "original_answer": rec["answer"][:120],
            "verdict": verdict_info.get("verdict", "unverifiable"),
            "ground_truth": verdict_info.get("ground_truth", ""),
            "reason": verdict_info.get("reason", ""),
        }


async def run_eval(sample: list[dict], concurrency: int, verbose: bool) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    tasks = [judge_one(r, sem, verbose) for r in sample]
    return await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="LLM 裁判抽样评估")
    parser.add_argument("--progress", default="output/results.csv.progress.jsonl",
                        help="结果进度文件路径")
    parser.add_argument("--sample", type=int, default=30, help="抽样题数")
    parser.add_argument("--out", default="eval_report.json", help="评估报告输出路径")
    parser.add_argument("--concurrency", type=int, default=4, help="裁判并发数")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    records = load_results(args.progress)
    records.sort(key=lambda r: int(r["key"]))
    sample = stratified_sample(records, args.sample)
    print(f"从 {len(records)} 题中抽样 {len(sample)} 题（文档{sum(1 for r in sample if r.get('citations'))}/SQL{sum(1 for r in sample if not r.get('citations'))}）")

    results = asyncio.run(run_eval(sample, args.concurrency, args.verbose))

    # 统计
    from collections import Counter
    verdicts = Counter(r["verdict"] for r in results)
    n = len(results)
    print(f"\n===== 评估结果（{n} 题）=====")
    for v in ("correct", "partially_correct", "incorrect", "unverifiable"):
        c = verdicts.get(v, 0)
        print(f"  {v}: {c} ({c/n*100:.1f}%)")
    correct_rate = (verdicts.get("correct", 0) + verdicts.get("partially_correct", 0) * 0.5) / n * 100
    print(f"  估算准确率: {correct_rate:.1f}%")

    # 明细
    print("\n--- 明细 ---")
    for r in results:
        print(f"[{r['id']}] {r['verdict']} | 真值={r['ground_truth'][:50]} | "
              f"答案={r['original_answer'][:50].replace(chr(10),' ')}")

    # 写报告
    report = {"total": n, "verdicts": dict(verdicts),
              "accuracy_estimate": round(correct_rate, 1), "details": results}
    pathlib.Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {args.out}")


if __name__ == "__main__":
    main()
