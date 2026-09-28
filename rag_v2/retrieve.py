"""向量库加载与检索。向量库实现集中在本模块，便于替换（Chroma -> FAISS 约 10 行）。"""

import math
import re

from langchain_core.documents import Document
from langchain_chroma import Chroma

from .config import get_embeddings, vectorstore_dir

COLLECTION_COUNT_HINT = "向量库为空，请先运行: python -m rag_v2.ingest"
GET_BATCH = 5000  # 全量分页大小，规避大库 get() 的 SQLite 变量数上限
KEYWORD_K = 6

# 公司 -> 该公司全部 chunk 的关键词检索缓存（进程内）
_company_chunk_cache: dict[str, list[dict]] = {}
_ALNUM = re.compile(r"[0-9a-zA-Z]+")
# 检索/匹配中低信息量的泛词（公司组织形式、常见动词等）
_STOPWORDS = {
    "公司", "有限", "股份", "集团", "报告", "报告期", "分别", "多少", "如何", "怎么",
    "请问", "我们", "他们", "以及", "根据", "相关", "情况", "数据", "比例", "金额",
}


def load_vectorstore() -> Chroma:
    vs = Chroma(persist_directory=vectorstore_dir(), embedding_function=get_embeddings())
    if vs._collection.count() == 0:
        raise RuntimeError(COLLECTION_COUNT_HINT)
    return vs


def get_all_metadatas() -> list[dict]:
    """分页拉取全部 metadata（库较大时一次 get 会触发 too many SQL variables）。"""
    vs = load_vectorstore()
    total = vs._collection.count()
    metas = []
    for offset in range(0, total, GET_BATCH):
        res = vs._collection.get(include=["metadatas"], limit=GET_BATCH, offset=offset)
        metas.extend(res["metadatas"])
    return metas


def get_all_ids() -> set[str]:
    """分页拉取全部 id，用于入库增量去重（空库返回空集，不报错）。"""
    vs = Chroma(persist_directory=vectorstore_dir(), embedding_function=get_embeddings())
    total = vs._collection.count()
    ids = []
    for offset in range(0, total, GET_BATCH):
        res = vs._collection.get(include=[], limit=GET_BATCH, offset=offset)
        ids.extend(res["ids"])
    return set(ids)


def make_retriever(company: str | None = None, k: int = 6):
    """构造 retriever；company 非空时按 metadata 过滤（替代旧链路的实体匹配）。"""
    vs = load_vectorstore()
    search_kwargs = {"k": k}
    if company:
        search_kwargs["filter"] = {"company": company}
    return vs.as_retriever(search_kwargs=search_kwargs)


def search(query: str, company: str | None = None, k: int = 6):
    """执行检索，返回 Document 列表。"""
    return make_retriever(company=company, k=k).invoke(query)


def format_docs(docs) -> str:
    """将检索结果格式化为带来源编号的上下文，供 agent 引用。"""
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        cid = doc.metadata.get("chunk_id", "")
        company = doc.metadata.get("company", "")
        parts.append(
            f"[{i}] 来源: {src}#{cid} | 公司: {company}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(parts)


def list_companies() -> list[str]:
    """返回库中所有公司名（去重）。"""
    return sorted({m.get("company", "") for m in get_all_metadatas() if m.get("company")})


# ---- 关键词检索（BM25-lite，汉字二元组 + 字母数字词） ----
# 语义检索对"问题直述句式"排序较差（实测答案 chunk 可排到 15 名开外），
# 本函数提供词面检索兜底，专找精确指标/数字/专有名词所在 chunk。

def _features(text: str) -> list[str]:
    """文本 -> 特征列表：连续汉字段生成二元组，字母数字串作为单词。"""
    feats = []
    for seg in re.findall(r"[一-鿿]+", text):
        # 去掉常见泛词后切二元组
        cleaned = seg
        for w in _STOPWORDS:
            cleaned = cleaned.replace(w, " ")
        for piece in cleaned.split():
            if len(piece) == 1:
                feats.append(piece)
            else:
                feats.extend(piece[i:i + 2] for i in range(len(piece) - 1))
    feats.extend(w.lower() for w in _ALNUM.findall(text))
    return feats


def _load_company_index(company: str):
    """加载某公司全部 chunk 并构建 BM25 索引（进程内缓存）。"""
    if company in _company_chunk_cache:
        return _company_chunk_cache[company]
    vs = load_vectorstore()
    col = vs._collection
    total = col.count()
    docs, metas, ids = [], [], []
    # where 过滤 + 分页（单公司约 500 chunk，一批足够，保险起见仍分页）
    offset = 0
    while True:
        res = col.get(where={"company": company}, include=["documents", "metadatas"],
                      limit=GET_BATCH, offset=offset)
        if not res["ids"]:
            break
        ids.extend(res["ids"])
        docs.extend(res["documents"])
        metas.extend(res["metadatas"])
        offset += GET_BATCH
        if len(ids) >= total:
            break

    from collections import Counter
    n = len(docs)
    tf_list, df = [], Counter()
    for text in docs:
        tf = Counter(_features(text))
        tf_list.append(tf)
        df.update(tf.keys())
    idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    avgdl = sum(sum(tf.values()) for tf in tf_list) / max(n, 1) or 1.0
    index = {"docs": docs, "metas": metas, "tf": tf_list, "idf": idf, "avgdl": avgdl}
    _company_chunk_cache[company] = index
    return index


def keyword_search(query: str, company: str, k: int = KEYWORD_K) -> list[Document]:
    """在指定公司的 chunk 内做 BM25 词面检索。company 必填（先 list_companies 确认全称）。"""
    if not company:
        raise ValueError("关键词检索必须指定 company（公司全称）")
    idx = _load_company_index(company)
    from collections import Counter
    q_feats = set(_features(query))
    k1, b = 1.5, 0.75
    scores = []
    for i, tf in enumerate(idx["tf"]):
        score = 0.0
        dl = sum(tf.values()) or 1
        for t in q_feats:
            if t in tf:
                idf = idx["idf"].get(t, 0.0)
                score += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * dl / idx["avgdl"]))
        if score > 0:
            scores.append((score, i))
    scores.sort(reverse=True)
    return [
        Document(page_content=idx["docs"][i], metadata=idx["metas"][i])
        for _, i in scores[:k]
    ]
