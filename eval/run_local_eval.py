"""本地评测：用官方评分逻辑给自建验证集打分。

流程：
1. 复用 rag_v2.ask 批跑 eval/standard_answer.jsonl -> eval/val_results.csv（断点续跑）
2. 转为官方提交格式 eval/submit_result.jsonl（id/answer）
3. 加载本地 text2vec-base-chinese 注入官方 evaluate 模块，调用官方 evaluate() 打分
   （semantic 维度的 text2vec 与线上一致；answer_term 子串匹配与 F1 完全一致）
4. 额外输出 confirmed 子集均分（review 题金标可能不准，单独看）

用法:
    python -m eval.run_local_eval                 # 跑答案 + 打分
    python -m eval.run_local_eval --skip-run      # 只打分（submit_result.jsonl 已存在）
    python -m eval.run_local_eval --download-model # 只下载 text2vec 模型
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
APP_ROOT = EVAL_DIR.parent
MODEL_DIR = EVAL_DIR / "models" / "text2vec-base-chinese"
STANDARD_PATH = EVAL_DIR / "standard_answer.jsonl"
SUBMIT_PATH = EVAL_DIR / "submit_result.jsonl"
VAL_RESULTS_CSV = EVAL_DIR / "val_results.csv"
DETAIL_PATH = EVAL_DIR / "evaluate_result_detail.jsonl"
SCORE_PATH = EVAL_DIR / "local_score.json"

MODEL_REPO = "shibing624/text2vec-base-chinese"


def download_model():
    """从 HF 镜像下载官方语义模型到 eval/models/。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # hf-mirror 不代理 xet CAS，会 401
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=MODEL_REPO, local_dir=str(MODEL_DIR))
    print(f"模型已下载: {path}")


def run_answers(concurrency: int):
    from rag_v2.ask import run_batch
    run_batch(str(STANDARD_PATH), str(VAL_RESULTS_CSV), None, False, concurrency)
    with open(VAL_RESULTS_CSV, newline="", encoding="utf-8-sig") as f, \
            open(SUBMIT_PATH, "w", encoding="utf-8") as out:
        for r in csv.DictReader(f):
            out.write(json.dumps({"id": int(r["id"]), "answer": r["答案"]},
                                 ensure_ascii=False) + "\n")
    print(f"提交文件已写入: {SUBMIT_PATH}")


def score():
    from text2vec import Similarity
    from . import evaluate as ev

    if not MODEL_DIR.exists():
        sys.exit(f"模型不存在: {MODEL_DIR}\n先执行: python -m eval.run_local_eval --download-model")

    # 注入官方模块全局 sim_model，与线上评测同模型
    ev.sim_model = Similarity(str(MODEL_DIR), max_seq_length=256)

    standard = ev.read_jsonl(str(STANDARD_PATH))
    submit = ev.read_jsonl(str(SUBMIT_PATH))
    standard.sort(key=lambda s: s["id"])
    submit.sort(key=lambda s: s["id"])
    if [s["id"] for s in standard] != [s["id"] for s in submit]:
        sys.exit("standard 与 submit 的 id 未对齐，请检查 submit_result.jsonl")

    os.chdir(EVAL_DIR)  # 官方 evaluate 把明细写到 ./evaluate_result_detail.jsonl
    scores = ev.evaluate(standard, submit)
    ev.report_score(scores, str(SCORE_PATH))

    # confirmed 子集均分（review/failed 题的金标可信度低）
    status_by_id = {s["id"]: s.get("status", "confirmed") for s in standard}
    detail = json.loads(DETAIL_PATH.read_text(encoding="utf-8"))
    confirmed = [d["score"] for d in detail if status_by_id.get(d["id"]) == "confirmed"]
    if confirmed:
        scores["score_confirmed_only"] = round(sum(confirmed) / len(confirmed) * 100, 2)
        scores["confirmed_n"] = len(confirmed)

    print("Scores:", json.dumps(scores, ensure_ascii=False, indent=2))
    return scores


def main():
    parser = argparse.ArgumentParser(description="自建验证集本地评测")
    parser.add_argument("--skip-run", action="store_true", help="跳过答题，直接打分")
    parser.add_argument("--download-model", action="store_true", help="仅下载 text2vec 模型")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    if args.download_model:
        download_model()
        return
    if not args.skip_run:
        run_answers(args.concurrency)
    score()


if __name__ == "__main__":
    main()
