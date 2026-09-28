"""自建验证集金标生成。

从 question.json 分层抽样（60 SQL + 40 文档），自动生成官方评测格式标准答案：
- SQL 题（数据查询）：LLM 独立编写 SQL -> 只读执行 -> LLM 根据结果撰写答案
- 文档题（文本理解）：汇集 results.csv 引用 chunk + 语义/关键词重检索 -> LLM 基于原文撰写答案
- 每题从答案中抽取 answer_term 关键数据点，经官方日期标准化后做子串校验
- 与既有 results.csv 答案由 LLM 交叉判定：一致 confirmed，不一致 review（保留入集）

用法: python -m eval.build_valset [--sql-n 60] [--doc-n 40] [--concurrency 6]
输出: eval/standard_answer.jsonl（官方格式 + status/judge_reason/sql 等附加字段）
进度: eval/valset_progress.jsonl（断点续跑，failed 记录重跑时自动重试）
"""

import argparse
import csv
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rag_v2.config import APP_ROOT, get_chat_model
from rag_v2 import retrieve
from rag_v2.sql_tools import TABLE_NOTES, _connect, _validate
from .evaluate import standardize_extended_date_formats

_TABLES = [
    "基金基本信息", "基金股票持仓明细", "基金债券持仓明细", "基金可转债持仓明细",
    "基金日行情表", "A股票日行情表", "港股票日行情表", "A股公司行业划分表",
    "基金规模变动表", "基金份额持有人结构",
]
_SCHEMA_CACHE: str | None = None

EVAL_DIR = Path(__file__).resolve().parent
QUESTION_PATH = APP_ROOT / "bs_challenge_financial_14b_dataset" / "question.json"
RESULTS_CSV = APP_ROOT / "output" / "results.csv"
if not RESULTS_CSV.exists() and (APP_ROOT / "results.csv").exists():
    RESULTS_CSV = APP_ROOT / "results.csv"  # 兼容旧布局（根目录）
STANDARD_PATH = EVAL_DIR / "standard_answer.jsonl"
PROGRESS_PATH = EVAL_DIR / "valset_progress.jsonl"

_GOLD_ROWS = 200  # 金标 SQL 执行行数上限（高于工具的 30，避免截断）
_MAX_TERMS = 8
_EVIDENCE_CHUNKS = 8
_CHUNK_CHARS = 900

_SQL_GEN_PROMPT = """你是金融数据库专家。根据问题编写一条 SQLite SELECT 查询。

数据库全部表结构（含示例行）：
{schema}

通用规则：
- 只输出 JSON：{{"sql": "..."}}，单条 SELECT/WITH，无其他文字
- 代码类值加引号保留前导零，如 股票代码='002244'
- 含括号列名用双引号，如 "收盘价(元)"；列名必须与上面表结构完全一致
- 明细查询加 LIMIT 20；聚合查询无需 LIMIT
- 严格按题目口径（行业分类标准、日期、是否包含边界值）

问题：{question}"""

_SQL_FIX_PROMPT = """上次 SQL 执行失败，请参考表结构修正后重新输出 JSON：{{"sql": "..."}}

数据库全部表结构（含示例行）：
{schema}

问题：{question}
失败 SQL：{sql}
错误信息：{error}"""

_SQL_ANSWER_PROMPT = """根据 SQL 查询结果回答问题。

问题：{question}
SQL：{sql}
查询结果：
{sql_result}

要求：
1. 用简洁中文直接作答（1-3 句），关键数值必须出现在答案原文中
2. 严格遵守题目格式要求（如"百分数保留两位小数"则写 12.34%）
3. 从答案中抽取关键数据点 terms：数字、代码、名称、日期等，每项必须是答案原文的连续子串，不超过 8 项
4. 只输出 JSON：{{"answer": "...", "terms": ["...", "..."]}}"""

_DOC_ANSWER_PROMPT = """根据给定的招股说明书资料片段回答问题。

问题：{question}

资料片段：
{evidence}

要求：
1. 严格依据资料内容作答，不得编造；用简洁中文直接作答（1-3 句）
2. 关键事实（名称、数字、日期等）必须与资料原文一致并出现在答案中
3. 从答案中抽取关键数据点 terms：每项必须是答案原文的连续子串，不超过 8 项
4. 若资料不足以回答，输出 {{"insufficient": true}}
5. 只输出 JSON：{{"answer": "...", "terms": ["...", "..."]}}"""

_JUDGE_PROMPT = """判断两个答案对同一问题的核心事实/数据是否一致。

问题：{question}
答案A：{answer_a}
答案B：{answer_b}

判定规则：
- 数值写法差异（千分位、单位换算如 2.6亿股 vs 2627492624股、日期格式、百分号有无）视为一致
- 核心数值不同、结论相反、张冠李戴视为不一致
- 只输出 JSON：{{"consistent": true/false, "reason": "一句话原因"}}"""

_TERMS_FROM_ANSWER_PROMPT = """从下面答案中抽取对评分关键的数据点。

问题：{question}
答案：{answer}

要求：抽取数字、代码、名称、日期等关键数据点，每项必须是答案原文的连续子串，不超过 8 项。
只输出 JSON：{{"terms": ["...", "..."]}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_write_lock = threading.Lock()


def _chat_json(llm, prompt: str, retries: int = 2) -> dict:
    """调用 LLM 并解析 JSON 输出，失败重试。"""
    last_err = None
    for _ in range(retries):
        try:
            resp = llm.invoke(prompt).content.strip()
            m = _JSON_RE.search(resp)
            return json.loads(m.group(0))
        except Exception as e:
            last_err = e
            prompt += "\n\n上次输出不是合法 JSON，请只输出 JSON 对象。"
    raise RuntimeError(f"LLM JSON 解析失败: {last_err}")


def _load_questions() -> dict[int, str]:
    questions = {}
    for line in QUESTION_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            questions[int(obj["id"])] = obj["question"].strip()
    return questions


def _load_results() -> dict[int, dict]:
    """results.csv -> {id: {answer, citations}}，引用非空判为文档题。"""
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"未找到首跑结果 {RESULTS_CSV}。请先运行批量问答生成 "
            f"output/results.csv（python -m rag_v2.ask --batch ... --out output/results.csv），"
            f"或将既有 results.csv 放到该路径后重试。"
        )
    out = {}
    with open(RESULTS_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cits = [c for c in (r.get("引用来源") or "").split(";") if c.strip()]
            out[int(r["id"])] = {"answer": r["答案"], "citations": cits}
    return out


def _sample(questions: dict, results: dict, sql_n: int, doc_n: int) -> list[dict]:
    sql_ids = sorted(i for i in questions if not results[i]["citations"])
    doc_ids = sorted(i for i in questions if results[i]["citations"])
    rng = random.Random(42)
    picked = [(i, "数据查询") for i in rng.sample(sql_ids, min(sql_n, len(sql_ids)))]
    picked += [(i, "文本理解") for i in rng.sample(doc_ids, min(doc_n, len(doc_ids)))]
    return [
        {"id": i, "type": t, "question": questions[i],
         "prev_answer": results[i]["answer"], "citations": results[i]["citations"]}
        for i, t in sorted(picked)
    ]


# ---------- 文档题证据收集（主线程串行，避免多线程访问 Chroma） ----------

def _fetch_chunk(source: str, cid: str) -> dict | None:
    vs = retrieve.load_vectorstore()
    res = vs._collection.get(
        where={"$and": [{"chunk_id": cid}, {"source": source}]},
        include=["documents", "metadatas"], limit=2,
    )
    if not res["ids"]:
        return None
    return {"content": res["documents"][0], "meta": res["metadatas"][0]}


def _collect_evidence(item: dict) -> str:
    """引用 chunk 优先，补充语义检索与公司内关键词检索，去重后编号拼接。"""
    seen, chunks, company = set(), [], None

    def add(content: str, meta: dict):
        key = (meta.get("source", ""), meta.get("chunk_id", ""))
        if key in seen or len(chunks) >= _EVIDENCE_CHUNKS:
            return
        seen.add(key)
        chunks.append((content, meta))

    for cit in item["citations"]:
        if "#" not in cit:
            continue
        source, cid = cit.rsplit("#", 1)
        hit = _fetch_chunk(source, cid)
        if hit:
            if company is None:
                company = hit["meta"].get("company")
            add(hit["content"], hit["meta"])

    try:
        for doc in retrieve.search(item["question"], company=company, k=6):
            add(doc.page_content, doc.metadata)
    except Exception:
        pass
    if company:
        try:
            for doc in retrieve.keyword_search(item["question"], company, k=4):
                add(doc.page_content, doc.metadata)
        except Exception:
            pass

    parts = []
    for i, (content, meta) in enumerate(chunks, 1):
        text = content.strip()[:_CHUNK_CHARS]
        parts.append(f"[{i}] 来源: {meta.get('source', '')} | 公司: {meta.get('company', '')}\n{text}")
    return "\n\n".join(parts)


# ---------- 金标生成 ----------

def _schema_context() -> str:
    """全部 10 张表的列定义 + 1 行示例，进程内缓存（线程安全：失败不留缓存）。"""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE:
        return _SCHEMA_CACHE
    con = _connect()
    try:
        parts = []
        for t in _TABLES:
            cols = [f"{c[1]}" for c in con.execute(f'PRAGMA table_info("{t}")').fetchall()]
            row = con.execute(f'SELECT * FROM "{t}" LIMIT 1').fetchone()
            sample = ""
            if row:
                vals = [str(v)[:20] if v is not None else "" for v in row]
                sample = "\n  示例: " + " | ".join(vals)
            note = f"\n  注意: {TABLE_NOTES[t]}" if t in TABLE_NOTES else ""
            parts.append(f"- {t}({', '.join(cols)}){note}{sample}")
        ctx = "\n".join(parts)
        _SCHEMA_CACHE = ctx
        return ctx
    finally:
        con.close()


def _run_gold_sql(sql: str) -> tuple[str | None, str | None]:
    """执行金标 SQL，返回 (结果文本, 错误)。"""
    err = _validate(sql)
    if err:
        return None, err
    try:
        con = _connect()
        try:
            cur = con.execute(sql)
            rows = cur.fetchmany(_GOLD_ROWS)
            if not rows:
                return None, "查询结果为空（0 行）"
            cols = [d[0] for d in cur.description]
            lines = [" | ".join(cols)]
            lines += [" | ".join("" if v is None else str(v) for v in r) for r in rows]
            return "\n".join(lines), None
        finally:
            con.close()
    except Exception as e:
        return None, str(e)


def _valid_terms(answer: str, terms: list) -> list[str]:
    """term 必须是（日期标准化后的）答案子串，去重并限量。"""
    std_answer = standardize_extended_date_formats(answer)
    out = []
    for t in terms:
        t = str(t).strip()
        if t and t not in out and standardize_extended_date_formats(t) in std_answer:
            out.append(t)
        if len(out) >= _MAX_TERMS:
            break
    return out


def _gold_for_sql(llm, question: str) -> dict:
    schema = _schema_context()
    obj = _chat_json(llm, _SQL_GEN_PROMPT.format(schema=schema, question=question))
    sql = obj["sql"].strip().rstrip(";")
    result, err = _run_gold_sql(sql)
    for _ in range(3):  # 失败让 LLM 看着表结构修，最多 3 轮
        if not err:
            break
        obj = _chat_json(llm, _SQL_FIX_PROMPT.format(
            schema=schema, question=question, sql=sql, error=err))
        sql = obj["sql"].strip().rstrip(";")
        result, err = _run_gold_sql(sql)
    if err:
        return {"status": "failed", "fail_reason": f"SQL 执行失败: {err}", "sql": sql,
                "answer": "", "answer_term": []}
    ans = _chat_json(llm, _SQL_ANSWER_PROMPT.format(
        question=question, sql=sql, sql_result=result))
    answer = ans.get("answer", "").strip()
    return {"status": "ok", "sql": sql, "answer": answer,
            "answer_term": _valid_terms(answer, ans.get("terms", []))}


def _gold_for_doc(llm, question: str, evidence: str) -> dict:
    if not evidence:
        return {"status": "failed", "fail_reason": "无可用证据 chunk", "answer": "", "answer_term": []}
    ans = _chat_json(llm, _DOC_ANSWER_PROMPT.format(question=question, evidence=evidence))
    if ans.get("insufficient"):
        return {"status": "failed", "fail_reason": "证据不足（LLM 判定）", "answer": "", "answer_term": []}
    answer = ans.get("answer", "").strip()
    return {"status": "ok", "answer": answer,
            "answer_term": _valid_terms(answer, ans.get("terms", []))}


def _judge(llm, question: str, gold: str, prev: str) -> tuple[str, str]:
    try:
        obj = _chat_json(llm, _JUDGE_PROMPT.format(question=question, answer_a=gold, answer_b=prev))
        return ("confirmed" if obj.get("consistent") else "review"), obj.get("reason", "")
    except Exception as e:
        return "review", f"判定失败: {e}"


def _process_one(item: dict, evidence: str | None) -> dict:
    llm = get_chat_model()
    try:
        if item["type"] == "数据查询":
            gold = _gold_for_sql(llm, item["question"])
        else:
            gold = _gold_for_doc(llm, item["question"], evidence or "")
    except Exception as e:
        gold = {"status": "failed", "fail_reason": str(e), "answer": "", "answer_term": []}

    record = {
        "id": item["id"], "type": item["type"], "question": item["question"],
        "answer": gold["answer"], "answer_term": gold["answer_term"],
    }
    if gold.get("sql"):
        record["sql"] = gold["sql"]
    if gold["status"] == "failed":
        record["status"] = "failed"
        record["judge_reason"] = gold.get("fail_reason", "")
    else:
        status, reason = _judge(llm, item["question"], gold["answer"], item["prev_answer"])
        record["status"] = status
        record["judge_reason"] = reason
    return record


def main():
    parser = argparse.ArgumentParser(description="自建验证集金标生成")
    parser.add_argument("--sql-n", type=int, default=60)
    parser.add_argument("--doc-n", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--fallback-prev", action="store_true",
                        help="不再重试 failed，直接用首跑 results.csv 答案兜底（标记 review）")
    args = parser.parse_args()

    questions, results = _load_questions(), _load_results()
    items = _sample(questions, results, args.sql_n, args.doc_n)
    print(f"抽样 {len(items)} 题（SQL {sum(1 for i in items if i['type'] == '数据查询')}，"
          f"文档 {sum(1 for i in items if i['type'] == '文本理解')}）")

    done = {}
    if PROGRESS_PATH.exists():
        for line in PROGRESS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") != "failed" or args.fallback_prev:
                    done[rec["id"]] = rec  # fallback 模式保留 failed 记录待兜底
    todo = [it for it in items if it["id"] not in done]
    print(f"已完成 {len(done)}，本次待生成 {len(todo)}")

    # 文档题证据收集放主线程串行（Chroma 读取非线程安全），LLM 调用再走线程池
    evidences = {}
    for it in todo:
        if it["type"] == "文本理解":
            evidences[it["id"]] = _collect_evidence(it)

    if todo:
        finished = 0
        with open(PROGRESS_PATH, "a", encoding="utf-8") as pf:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(_process_one, it, evidences.get(it["id"])): it for it in todo}
                for fut in as_completed(futures):
                    it = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as e:  # 单个失败不拖垮整批
                        rec = {"id": it["id"], "type": it["type"], "question": it["question"],
                               "answer": "", "answer_term": [], "status": "failed",
                               "judge_reason": f"未捕获异常: {e}"}
                    with _write_lock:
                        pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        pf.flush()
                    done[rec["id"]] = rec
                    finished += 1
                    if finished % 10 == 0:
                        print(f"进度 {finished}/{len(todo)}", flush=True)

    records = [done[it["id"]] for it in items if it["id"] in done]

    if args.fallback_prev:
        # failed 记录兜底：用首跑 results.csv 答案作金标，抽 terms，标记 review
        prev_by_id = {it["id"]: it["prev_answer"] for it in items}
        llm = get_chat_model()
        for rec in records:
            prev = prev_by_id.get(rec["id"], "")
            if rec["status"] != "failed" or not prev or prev.startswith("ERROR"):
                continue
            reason = rec.get("judge_reason", "")[:60]
            try:
                obj = _chat_json(llm, _TERMS_FROM_ANSWER_PROMPT.format(
                    question=rec["question"], answer=prev))
                terms = _valid_terms(prev, obj.get("terms", []))
            except Exception:
                terms = []
            rec.update({"answer": prev, "answer_term": terms, "status": "review",
                        "judge_reason": f"金标生成失败（{reason}），回退首跑答案"})
            print(f"兜底: id={rec['id']} terms={len(terms)}")

    records.sort(key=lambda r: r["id"])
    with open(STANDARD_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    stats = Counter(r["status"] for r in records)
    print(f"已写入 {STANDARD_PATH}（{len(records)} 题）: {dict(stats)}")


if __name__ == "__main__":
    main()
