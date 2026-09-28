"""结构化数据 SQL 工具：让 agent 直接查询大赛 SQLite 金融库（10 张表）。

安全约束:
    - 只读连接（URI mode=ro）
    - 仅允许单条 SELECT/WITH 语句
    - 返回行数上限 MAX_ROWS，超出截断
    - progress handler 超时中断，提示 agent 收窄过滤条件
"""

import re
import sqlite3
import time
from pathlib import Path

from langchain_core.tools import tool

from .config import db_path

MAX_ROWS = 30
QUERY_TIMEOUT_S = 60

_SELECT_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

# 每张表补充说明（日期格式等易错点），供 list_db_tables 返回
TABLE_NOTES = {
    "基金规模变动表": "公告日期/截止日期为 'YYYY-MM-DD HH:MM:SS' 格式",
    "基金份额持有人结构": "公告日期/截止日期为 'YYYY-MM-DD HH:MM:SS' 格式",
}

_TABLE_SUMMARY = """数据库共 10 张表（均为中文列名）：
- 基金基本信息: 基金代码/全称/简称/管理人/托管人/基金类型/成立日期/到期日期/管理费率/托管费率
- 基金股票持仓明细: 基金代码/简称/持仓日期/股票代码/股票名称/数量/市值/市值占基金资产净值比/第N大重仓股/所在证券市场/所属国家(地区)/报告类型
- 基金债券持仓明细: 基金代码/简称/持仓日期/债券类型/债券名称/持债数量/持债市值/持债市值占基金资产净值比/第N大重仓股/所在证券市场/所属国家(地区)/报告类型
- 基金可转债持仓明细: 基金代码/简称/持仓日期/对应股票代码/债券名称/数量/市值/市值占基金资产净值比/第N大重仓股/所在证券市场/所属国家(地区)/报告类型
- 基金日行情表: 基金代码/交易日期/单位净值/复权单位净值/累计单位净值/资产净值
- A股票日行情表: 股票代码/交易日/昨收盘(元)/今开盘(元)/最高价(元)/最低价(元)/收盘价(元)/成交量(股)/成交金额(元)
- 港股票日行情表: 股票代码/交易日/昨收盘(元)/今开盘(元)/最高价(元)/最低价(元)/收盘价(元)/成交量(股)/成交金额(元)
- A股公司行业划分表: 股票代码/交易日期/行业划分标准(如'中信行业分类'/'申万行业分类')/一级行业名称/二级行业名称
- 基金规模变动表: 基金代码/简称/公告日期/截止日期/期初总份额/总申购份额/总赎回份额/期末总份额/定期报告所属年度/报告类型
- 基金份额持有人结构: 基金代码/简称/公告日期/截止日期/机构持有份额/机构占比/个人持有份额/个人占比/定期报告所属年度/报告类型

通用规则: 股票/基金代码是 TEXT 且保留前导零，查询必须加引号（如 股票代码='002244'）；
日期多为 'YYYYMMDD' 字符串。完整列名与示例数据用 get_table_schema 查看。"""


def _connect() -> sqlite3.Connection:
    uri = Path(db_path()).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    deadline = time.time() + QUERY_TIMEOUT_S
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 100_000)
    return con


def _validate(sql: str) -> str | None:
    stripped = sql.strip().rstrip(";").strip()
    if not _SELECT_RE.match(stripped):
        return "仅允许单条 SELECT/WITH 查询语句。"
    if ";" in stripped:
        return "不允许包含多个语句（检测到内部分号）。"
    return None


def _format_rows(cur: sqlite3.Cursor, rows: list[tuple]) -> str:
    cols = [d[0] for d in cur.description]
    head = " | ".join(cols)
    lines = [head, "-" * min(len(head), 80)]
    for r in rows[:MAX_ROWS]:
        lines.append(" | ".join("" if v is None else str(v) for v in r))
    if len(rows) > MAX_ROWS:
        lines.append(f"...（仅显示前 {MAX_ROWS} 行，共 {len(rows)} 行）")
    lines.append(f"共 {len(rows)} 行")
    return "\n".join(lines)


@tool
def list_db_tables() -> str:
    """列出可查询的 SQLite 金融数据库的全部表及列摘要。回答股票、基金、行业等结构化数据问题前先调用本工具。"""
    return _TABLE_SUMMARY


@tool
def get_table_schema(table_name: str) -> str:
    """查看指定表的完整列定义和 3 行示例数据，用于确认列名写法与数据格式。

    Args:
        table_name: 表名，必须是 list_db_tables 列出的表之一。
    """
    try:
        con = _connect()
    except Exception as e:
        return f"数据库连接失败: {e}"
    try:
        cols = con.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        if not cols:
            return f"表 {table_name} 不存在。请用 list_db_tables 查看可用表。"
        col_lines = [f"- {c[1]} ({c[2]})" for c in cols]
        note = f"\n注意: {TABLE_NOTES[table_name]}" if table_name in TABLE_NOTES else ""
        sample = con.execute(f'SELECT * FROM "{table_name}" LIMIT 3').fetchall()
        cur = con.execute(f'SELECT * FROM "{table_name}" LIMIT 0')
        return (f"表 {table_name} 列定义:\n" + "\n".join(col_lines) + note
                + "\n示例数据:\n" + _format_rows(cur, sample))
    except Exception as e:
        return f"查询表结构失败: {e}"
    finally:
        con.close()


@tool
def run_sql(sql_query: str) -> str:
    """在金融数据库上执行只读 SELECT 查询并返回结果表格。

    规则:
    - 只能单条 SELECT/WITH 语句；必须加 LIMIT 且尽量利用 股票代码/日期 索引过滤
    - 含括号等特殊字符的列名用双引号包裹，如 "收盘价(元)"
    - 代码类值必须加引号保留前导零，如 股票代码='002244'
    - 聚合结果通常一行，无需 LIMIT；明细查询 LIMIT 不超过 20
    """
    err = _validate(sql_query)
    if err:
        return err
    try:
        con = _connect()
    except Exception as e:
        return f"数据库连接失败: {e}"
    try:
        cur = con.execute(sql_query)
        rows = cur.fetchmany(MAX_ROWS + 1)
        if not rows:
            return "查询结果为空（0 行）。请检查列名、日期格式或过滤条件后重试。"
        return _format_rows(cur, rows)
    except sqlite3.OperationalError as e:
        msg = str(e)
        if "interrupted" in msg.lower():
            return ("查询超时（>60s）已中断。请收窄过滤条件（指定股票代码/日期区间）"
                    "或改用聚合查询后重试。")
        return f"SQL 执行失败: {msg}。请检查表名/列名（可用 get_table_schema 确认）后重试。"
    except Exception as e:
        return f"SQL 执行失败: {e}"
    finally:
        con.close()


SQL_TOOLS = [list_db_tables, get_table_schema, run_sql]
