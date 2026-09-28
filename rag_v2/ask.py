"""问答 CLI。

单问:  python -m rag_v2.ask "青洲银行2022年营业收入是多少？"
批处理: python -m rag_v2.ask --batch question.json --out results.csv
        输入支持 .jsonl/.json（每行 {"id": n, "question": "..."}）或含"问题"列的 CSV；
        并发由 --concurrency 控制；结果逐题落盘 <out>.progress.jsonl，中断后重跑自动续跑。
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

from .agent import aask, ask


def run_one(question: str, verbose: bool) -> dict:
    try:
        return ask(question, verbose=verbose)
    except Exception:
        # 批处理场景失败重试一次
        return ask(question, verbose=verbose)


def load_questions(path: str) -> list[dict]:
    """读取批量问题，统一返回 [{key, question}]。key 为题目 id（CSV 时为行号）。"""
    p = Path(path)
    if p.suffix.lower() in (".jsonl", ".json"):
        rows = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            obj = json.loads(line)
            q = (obj.get("question") or obj.get("问题") or "").strip()
            if q:
                rows.append({"key": str(obj.get("id", i)), "question": q})
        return rows
    with open(p, newline="", encoding="utf-8-sig") as f:
        return [
            {"key": str(i), "question": r["问题"].strip()}
            for i, r in enumerate(csv.DictReader(f))
            if (r.get("问题") or "").strip()
        ]


async def _run_one_async(item: dict, sem: asyncio.Semaphore, progress_file, counter: dict):
    async with sem:
        result = None
        err = None
        for _ in range(2):  # 失败重试一次
            try:
                result = await aask(item["question"])
                err = None
                break
            except Exception as e:
                err = e
        record = {
            "key": item["key"],
            "question": item["question"],
            "answer": result["answer"] if result else f"ERROR: {err}",
            "citations": result["citations"] if result else [],
            "steps": len(result["steps"]) if result else 0,
        }
        progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        progress_file.flush()
        counter["done"] += 1
        if counter["done"] % 10 == 0:
            print(f"进度 {counter['done']}/{counter['total']}", flush=True)


def run_batch(input_path: str, out_path: str, max_questions: int | None,
              verbose: bool, concurrency: int):
    items = load_questions(input_path)
    if max_questions:
        items = items[:max_questions]

    # 断点续跑：读已完成的 key（ERROR 或递归保护话术视为未完成，自动重跑）
    progress_path = Path(str(out_path) + ".progress.jsonl")
    def _is_bad(ans: str) -> bool:
        return ans.startswith("ERROR") or ans.startswith("Sorry, need more steps")

    done = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if not _is_bad(str(rec.get("answer", ""))):
                    done[rec["key"]] = rec
    todo = [it for it in items if it["key"] not in done]
    if todo:
        # 重写进度文件，剔除将被重跑的 ERROR 记录
        with open(progress_path, "w", encoding="utf-8") as pf:
            for rec in done.values():
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"共 {len(items)} 个问题，已完成 {len(done)}，本次待跑 {len(todo)}")

    if todo:
        if verbose:
            # 并发下轨迹会交错，仅对第一题打印
            preview = todo[0]
            try:
                res = run_one(preview["question"], verbose=True)
                rec = {
                    "key": preview["key"], "question": preview["question"],
                    "answer": res["answer"], "citations": res["citations"],
                    "steps": len(res["steps"]),
                }
                with open(progress_path, "a", encoding="utf-8") as pf:
                    pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[preview["key"]] = rec
                todo = todo[1:]
            except Exception:
                pass

        async def _main():
            counter = {"done": 0, "total": len(todo)}
            sem = asyncio.Semaphore(concurrency)
            with open(progress_path, "a", encoding="utf-8") as pf:
                await asyncio.gather(*[_run_one_async(it, sem, pf, counter) for it in todo])

        asyncio.run(_main())
        # 重新加载全部进度
        done = {}
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec

    # 按原始顺序汇总写 CSV
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "问题", "答案", "引用来源", "工具调用次数"])
        for it in items:
            rec = done.get(it["key"])
            if rec:
                writer.writerow([rec["key"], rec["question"], rec["answer"],
                                 ";".join(rec["citations"]), rec["steps"]])
            else:
                writer.writerow([it["key"], it["question"], "ERROR: 未完成", "", 0])
    print(f"结果已写入: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="金融问答 CLI（文档 RAG + SQL）")
    parser.add_argument("question", nargs="?", help="单个问题")
    parser.add_argument("--batch", help="批量问题文件（.jsonl/.json 或含'问题'列的 CSV）")
    parser.add_argument("--out", default="results.csv", help="批处理输出路径")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=6, help="批处理并发数（默认 6）")
    parser.add_argument("--verbose", action="store_true", help="打印 agent 工具调用轨迹")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.batch, args.out, args.max_questions, args.verbose, args.concurrency)
    elif args.question:
        result = run_one(args.question, args.verbose)
        print(result["answer"])
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
