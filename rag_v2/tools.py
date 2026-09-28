"""提供给 agent 的检索工具（Tool Calling 的体现点）。"""

from langchain_core.tools import tool

from . import retrieve
from .sql_tools import SQL_TOOLS


@tool
def search_docs(query: str, company: str = "") -> str:
    """在文档库中检索与问题相关的段落。

    Args:
        query: 检索 query，可自由改写（如提取关键词、换同义表达）以提高命中率。
        company: 可选，公司全称。提供时只检索该公司文档；不确定时留空。
    """
    company = company.strip() or None
    docs = retrieve.search(query, company=company)
    if not docs:
        scope = f"公司「{company}」的" if company else ""
        return f"未检索到相关{scope}文档段落，可尝试改写 query 或不指定 company 重试。"
    return retrieve.format_docs(docs)


@tool
def list_companies() -> str:
    """列出文档库中收录的所有公司名称。问题涉及具体公司但不确定全称时先调用本工具。"""
    companies = retrieve.list_companies()
    if not companies:
        return "文档库中暂无公司信息。"
    return "文档库收录的公司：\n" + "\n".join(f"- {c}" for c in companies)


@tool
def search_docs_by_keyword(keywords: str, company: str) -> str:
    """在指定公司文档内按关键词做词面检索（BM25），用于语义检索搜不到时的兜底。

    适用：要找精确指标名称/数字/专有名词（如"人工成本占主营业务成本比例""在册员工总数"），
    而 search_docs 多次改写仍未命中。query 请直接堆砌问题中的核心词组（2-4 个），不要整句。

    Args:
        keywords: 核心关键词组合，如 "人工成本 主营业务成本 劳动力成本"。
        company: 公司全称（必填，先用 list_companies 确认）。
    """
    company = company.strip()
    if not company:
        return "关键词检索必须提供 company（公司全称），可先调用 list_companies。"
    docs = retrieve.keyword_search(keywords, company=company)
    if not docs:
        return f"公司「{company}」文档内无词面匹配段落，可更换关键词重试。"
    return retrieve.format_docs(docs)


ALL_TOOLS = [search_docs, search_docs_by_keyword, list_companies, *SQL_TOOLS]
