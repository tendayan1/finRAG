"""文档入库：加载 -> 切分 -> embedding -> 增量写入 Chroma。

用法:
    python -m rag_v2.ingest [--data-dir DIR] [--company-map CSV] [--reset]

company 元数据解析优先级:
    1. --company-map 指定的 CSV（含 csv文件名/公司名称 列，复用比赛映射文件）
    2. 文档首行 ``# 公司：XXX`` 约定
    3. 正文启发式抽取（招股说明书封面/名称锚点/高频机构名）
    4. 文件名 stem
"""

import argparse
import collections
import csv
import hashlib
import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import APP_ROOT, docs_dir, get_embeddings, vectorstore_dir
from .retrieve import get_all_ids

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]
INSERT_BATCH = 1000  # 分批 embed+写入，避免单次 payload/内存过大

# ---- 招股说明书公司名自动抽取（无 company-map 时兜底） ----
_ORG = r"[\u4e00-\u9fa5A-Za-z0-9]{2,20}?(?:集团股份有限公司|股份有限公司|有限责任公司)"
_ORG_EXACT_RE = re.compile(
    r"^[\u4e00-\u9fa5A-Za-z0-9（）()]{2,25}?(?:集团股份有限公司|股份有限公司|有限责任公司)$"
)
_NAME_ANCHORS = [
    re.compile(r"发行人(?:中文)?名称\s*[：:]\s*(" + _ORG + r")"),
    re.compile(r"企业名称\s*[：:]\s*(" + _ORG + r")"),
    re.compile(r"(?:^|\s|】)公司名称\s*[：:]\s*(" + _ORG + r")"),
    re.compile(r"中文名称\s*[：:]\s*(" + _ORG + r")"),
]
_ORG_ANY_RE = re.compile(_ORG)
# 频率兜底时排除的噪声/中介机构
_NAME_GARBAGE_RE = re.compile(r"变更为|变更设立|改制|更名|设立的|发起设立的|系由|原为|因本所")
_NAME_INTERMEDIARY_RE = re.compile(
    r"证券股份|证券有限|期货股份|会计师|律师|资产评估|资信评估|评级|交易所|登记结算|银行股份"
)


def _first_line_company(text: str) -> str | None:
    """文档前 15 行内单独成行的公司全称（常是封面）；带"证券"字样多为保荐机构，排除。"""
    for line in text.splitlines()[:15]:
        s = line.strip().replace(" ", "")
        if _ORG_EXACT_RE.match(s) and len(s) >= 8 and "证券" not in s:
            return s
    return None


def _maybe_lengthen(text: str, name: str) -> str:
    """处理 OCR 截断：若带 1-4 字汉字前缀的更长形式在文中出现得更多，则采用更长全称。

    例如封面 OCR 为"州光弘科技股份有限公司"，而正文 7 处均为"惠州光弘科技股份有限公司"。
    """
    standalone = len(re.findall(r"(?<![\u4e00-\u9fa5])" + re.escape(name), text))
    cands = collections.Counter(
        re.findall(r"[\u4e00-\u9fa5]{1,4}" + re.escape(name), text)
    )
    for longer, c in cands.most_common():
        if longer != name and c >= 2 and c > standalone and len(longer) <= len(name) + 4:
            return longer
    return name


def extract_company_from_text(text: str) -> str | None:
    """从招股说明书正文推断发行人全称：封面首行 -> 名称类锚点 -> 高频机构名兜底。"""
    first = _first_line_company(text)
    if first:
        return _maybe_lengthen(text, first)
    for rx in _NAME_ANCHORS:
        m = rx.search(text[:400000])
        if m:
            return m.group(1)
    count = collections.Counter(_ORG_ANY_RE.findall(text))
    for name, _ in count.most_common(20):
        if len(name) < 8 or _NAME_GARBAGE_RE.search(name) or _NAME_INTERMEDIARY_RE.search(name):
            continue
        return name
    return None



def load_company_map(csv_path: str) -> dict:
    """读取 csv文件名->公司名称 映射（复用 files/AF0_pdf_to_company.csv）。"""
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = (row.get("公司名称") or "").strip()
            key = (row.get("csv文件名") or "").strip()
            if name and key:
                # 兼容 xxx.PDF.csv / xxx.txt 等派生文件名
                for candidate in {key, key.replace(".PDF.csv", ""), Path(key).stem}:
                    mapping[candidate] = name
    return mapping


def resolve_company(doc_path: Path, text: str, company_map: dict) -> str:
    stem = doc_path.stem
    for candidate in (stem, doc_path.name):
        if candidate in company_map:
            return company_map[candidate]
    first_line = text.splitlines()[0].strip() if text.strip() else ""
    if first_line.startswith("# 公司："):
        return first_line.replace("# 公司：", "").strip()
    guessed = extract_company_from_text(text)
    if guessed:
        return guessed
    return stem


def load_documents(data_dir: str, company_map: dict):
    docs = []
    txt_loader_kwargs = {"encoding": "utf-8"}
    patterns = {"*.txt": TextLoader, "*.md": TextLoader}
    for pattern, loader_cls in patterns.items():
        loader = DirectoryLoader(
            data_dir, glob=f"**/{pattern}", loader_cls=loader_cls,
            loader_kwargs=txt_loader_kwargs, show_progress=True,
        )
        docs.extend(loader.load())
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader,
                                 show_progress=True)
        docs.extend(loader.load())
    except ImportError:
        if list(Path(data_dir).rglob("*.pdf")):
            print("检测到 .pdf 文件但未安装 pypdf，已跳过。pip install pypdf 后可支持。")

    for doc in docs:
        path = Path(doc.metadata["source"])
        doc.metadata["source"] = path.name
        doc.metadata["company"] = resolve_company(path, doc.page_content, company_map)
    return docs


def main():
    parser = argparse.ArgumentParser(description="将文档切分并写入向量库")
    parser.add_argument("--data-dir", default=docs_dir())
    parser.add_argument("--company-map", default=str(APP_ROOT / "files" / "AF0_pdf_to_company.csv"))
    parser.add_argument("--reset", action="store_true", help="清空后重建向量库")
    args = parser.parse_args()

    persist_dir = vectorstore_dir()
    if args.reset and Path(persist_dir).exists():
        shutil.rmtree(persist_dir)
        print(f"已清空向量库: {persist_dir}")

    company_map = {}
    if args.company_map and Path(args.company_map).exists():
        company_map = load_company_map(args.company_map)
        print(f"公司映射: {len(company_map)} 条 ({args.company_map})")

    if not Path(args.data_dir).is_dir():
        print(f"数据目录不存在: {args.data_dir}")
        return
    docs = load_documents(args.data_dir, company_map)
    if not docs:
        print(f"未在 {args.data_dir} 找到 .txt/.md/.pdf 文档")
        return
    print(f"加载文档 {len(docs)} 个")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=SEPARATORS,
    )
    chunks = splitter.split_documents(docs)
    # 稳定 id：内容哈希，保证增量去重
    ids = []
    for chunk in chunks:
        chunk.metadata["chunk_id"] = hashlib.sha1(
            (chunk.metadata["source"] + chunk.page_content).encode("utf-8")
        ).hexdigest()[:12]
        ids.append(chunk.metadata["chunk_id"])
    print(f"切分得到 {len(chunks)} 个 chunk")

    embeddings = get_embeddings()
    vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    # 去重：库中已有（分页拉取）+ 本次切分内部同 id（PDF 提取常产生重复目录/页眉片段）
    seen = get_all_ids()
    new_chunks, new_ids = [], []
    dup = 0
    for chunk, cid in zip(chunks, ids):
        if cid in seen:
            dup += 1
            continue
        seen.add(cid)
        new_chunks.append(chunk)
        new_ids.append(cid)
    print(f"待入库 {len(new_chunks)} 条，跳过已存在/重复 {dup} 条")

    if new_chunks:
        total_batches = (len(new_chunks) + INSERT_BATCH - 1) // INSERT_BATCH
        for bi, start in enumerate(range(0, len(new_chunks), INSERT_BATCH), 1):
            batch = new_chunks[start:start + INSERT_BATCH]
            batch_ids = new_ids[start:start + INSERT_BATCH]
            vs.add_documents(batch, ids=batch_ids)
            print(f"入库进度: {min(start + INSERT_BATCH, len(new_chunks))}/{len(new_chunks)} "
                  f"(批次 {bi}/{total_batches})，库中总计 {vs._collection.count()} 条", flush=True)
    print(f"入库完成: 新增 {len(new_chunks)} 条，跳过已存在 {len(chunks) - len(new_chunks)} 条，"
          f"库中总计 {vs._collection.count()} 条")


if __name__ == "__main__":
    main()
