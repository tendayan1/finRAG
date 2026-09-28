"""离线冒烟测试：用 FakeEmbeddings 验证 ingest -> retrieve -> tools 链路。

不调用任何外部 API，仅验证代码路径与数据结构正确性。
运行: python -m rag_v2._smoke_test
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from langchain_community.embeddings import FakeEmbeddings

fake = FakeEmbeddings(size=64)


def main():
    # ignore_cleanup_errors: Windows 下 Chroma 句柄释放有延迟，避免清理时报 WinError 32
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        vs_dir = str(Path(tmp) / "vs")
        with patch("rag_v2.config.get_embeddings", return_value=fake), \
             patch("rag_v2.ingest.get_embeddings", return_value=fake), \
             patch("rag_v2.ingest.vectorstore_dir", return_value=vs_dir), \
             patch("rag_v2.retrieve.get_embeddings", return_value=fake), \
             patch("rag_v2.retrieve.vectorstore_dir", return_value=vs_dir):

            from rag_v2 import ingest, retrieve, tools

            # 1. ingest（使用默认 sample_docs，跳过 company-map 缺失的影响）
            import sys
            sys.argv = ["ingest", "--company-map", "nonexistent.csv"]
            ingest.main()

            # 1b. 重复 ingest 验证增量去重（应新增 0 条）
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                ingest.main()
            assert "新增 0 条" in buf.getvalue(), f"增量去重失效: {buf.getvalue()}"
            print("  增量去重验证通过（重复 ingest 新增 0 条）")

            # 2. retrieve
            docs = retrieve.search("营业收入", k=3)
            assert docs, "检索结果为空"
            formatted = retrieve.format_docs(docs)
            assert "来源:" in formatted, "format_docs 缺少来源标注"
            companies = retrieve.list_companies()
            assert companies, "公司列表为空"
            print(f"  检索命中 {len(docs)} 条；库中公司: {companies}")

            # 3. company 过滤
            target = companies[0]
            filtered = retrieve.search("财务数据", company=target, k=3)
            assert all(d.metadata["company"] == target for d in filtered), \
                "company 过滤失效"
            print(f"  company={target} 过滤检索命中 {len(filtered)} 条，全部属于该公司")

            # 4. tools
            out = tools.search_docs.invoke({"query": "净利润", "company": ""})
            assert "来源:" in out
            out2 = tools.list_companies.invoke({})
            assert target in out2
            # 关键词检索兜底（必须带 company）
            out3 = tools.search_docs_by_keyword.invoke(
                {"keywords": "营业收入 净利润", "company": target})
            assert "来源:" in out3 and target in out3, f"关键词检索异常: {out3[:200]}"
            assert "必须提供 company" in tools.search_docs_by_keyword.invoke(
                {"keywords": "x", "company": ""})
            print("  工具调用正常（语义/关键词检索）")

            # 5. SQL 工具（离线：临时 SQLite 库，不触碰大赛库）
            import sqlite3 as _sq
            sql_db = Path(tmp) / "mini.db"
            _c = _sq.connect(sql_db)
            _c.execute('CREATE TABLE "A股票日行情表" (股票代码 TEXT, 交易日 TEXT, "收盘价(元)" REAL)')
            _c.execute("INSERT INTO \"A股票日行情表\" VALUES ('002244','20191220',4.67)")
            _c.commit()
            _c.close()
            from rag_v2 import sql_tools
            with patch("rag_v2.sql_tools.db_path", return_value=str(sql_db)):
                assert "A股票日行情表" in sql_tools.list_db_tables.invoke({})
                schema = sql_tools.get_table_schema.invoke({"table_name": "A股票日行情表"})
                assert "收盘价(元)" in schema and "4.67" in schema
                r = sql_tools.run_sql.invoke({
                    "sql_query": "select 股票代码, \"收盘价(元)\" from A股票日行情表 "
                                 "where 股票代码='002244' and 交易日='20191220'"})
                assert "4.67" in r, f"SQL 点查失败: {r}"
                assert "仅允许" in sql_tools.run_sql.invoke({"sql_query": "delete from t"})
                assert "不允许包含多个语句" in sql_tools.run_sql.invoke(
                    {"sql_query": "select 1; drop table t"})
            print("  SQL 工具正常（点查/安全校验通过）")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
