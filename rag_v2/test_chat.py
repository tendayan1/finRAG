"""Chat 模型最小连通测试：发一条消息验证 .env 配置的端点/密钥/模型可用。

运行: python -m rag_v2.test_chat
"""

from .config import chat_model_name, get_chat_model


def main():
    print(f"模型: {chat_model_name()}")
    try:
        reply = get_chat_model().invoke("请只回复四个字：连通正常")
    except Exception as e:
        print(f"连通失败: {e}")
        raise SystemExit(1)
    text = reply.content if hasattr(reply, "content") else str(reply)
    print(f"回复: {text.strip()}")
    print("CHAT CONNECTIVITY OK")


if __name__ == "__main__":
    main()
