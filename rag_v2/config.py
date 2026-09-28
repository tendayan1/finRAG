"""配置加载与 LLM / Embedding 客户端工厂。

所有外部服务均走 OpenAI 兼容 API，通过 .env 配置，可插拔
DeepSeek / Qwen(DashScope 兼容模式) / vLLM / Ollama 等后端。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

APP_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(APP_ROOT / ".env")

DEFAULT_DOCS_DIR = APP_ROOT / "data" / "sample_docs"
DEFAULT_DB_PATH = (
    APP_ROOT / "bs_challenge_financial_14b_dataset" / "dataset" / "博金杯比赛数据.db"
)


def vectorstore_dir() -> str:
    return os.environ.get("VECTORSTORE_DIR", str(APP_ROOT / "vectorstore"))


def db_path() -> str:
    return os.environ.get("DB_PATH", str(DEFAULT_DB_PATH))


def docs_dir() -> str:
    return os.environ.get("DOCS_DIR", str(DEFAULT_DOCS_DIR))


def chat_model_name() -> str:
    return os.environ.get("CHAT_MODEL", "deepseek-chat")


def embedding_model_name() -> str:
    model = os.environ.get("EMBEDDING_MODEL")
    if not model:
        raise RuntimeError(
            "未配置 EMBEDDING_MODEL。请在 .env 中设置支持 /embeddings 的模型，"
            "参见 .env.example。"
        )
    return model


def get_chat_model(**overrides) -> ChatOpenAI:
    """返回对话模型客户端。temperature=0 保证答案可复现。"""
    kwargs = {"model": chat_model_name(), "temperature": 0}
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)


def get_embeddings() -> OpenAIEmbeddings:
    """返回 embedding 客户端。不传 dimensions 以兼容各类 OpenAI 兼容端点。

    embedding 与 chat 可指向不同服务：设置 EMBEDDING_BASE_URL / EMBEDDING_API_KEY
    时优先使用，否则回落到 OPENAI_BASE_URL / OPENAI_API_KEY。
    """
    # check_embedding_ctx_length=False：OpenAI 兼容端点（DashScope 等）
    # 不接受 token 数组，直接发送原始文本
    # chunk_size=10：DashScope 单批 embedding 请求上限 10 条（默认 1000 会报 400）
    kwargs = {
        "model": embedding_model_name(),
        "check_embedding_ctx_length": False,
        "chunk_size": 10,
    }
    base_url = os.environ.get("EMBEDDING_BASE_URL")
    api_key = os.environ.get("EMBEDDING_API_KEY")
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)
