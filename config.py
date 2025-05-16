from pydantic import BaseModel, Field
from typing import Literal, Dict
import torch
import os
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI


class ModelConfig(BaseModel):
    """LLM模型配置"""
    title: str = Field(..., description="模型标题")
    model: str = Field(..., description="模型名称")
    provider: str = Field(..., description="模型提供商")
    apiBase: str = Field(..., description="API基础URL")
    apiKey: str = Field(..., description="API密钥")
    temperature: float = Field(0.8, description="采样温度")
    max_tokens: int = Field(2048, description="最大生成长度")
    model_type: str = Field("chat", description="模型类型：chat或completion")


class Config(BaseModel):
    # 设备配置
    DEVICE: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_MEMORY: int = 16  # GB，内存限制

    # 模型路径
    BERT_MODEL: str = "shibing624/bert4ner-base-chinese"  # 中文基础模型
    HF_MIRROR: str = "https://hf-mirror.com"  # 国内镜像
    SENTENCE_BERT: str = "shibing624/text2vec-base-chinese"  # 中文句向量模型
    CHINESE_STOPWORDS_PATH: str = "./utils/hit_stopwords.txt"

    # 数据库配置
    CHROMA_PERSIST_PATH: str = "chroma_db"
    KG_PERSIST_PATH: str = "chroma_db/knowledge_graph.gml"

    # 向量参数
    VECTOR_DIM: int = 768

    # LLM配置
    LLM_MODEL: ModelConfig = ModelConfig(
        title="qwen3:30b",
        model="qwen3:30b",
        provider="openai",
        apiBase="http://192.168.0.1:8000/version",
        apiKey="set_password",
        temperature=0.9,
        max_tokens=2048,
        model_type="chat"
    )


def setup_huggingface_mirror(config):
    """设置Hugging Face镜像源"""
    if config.HF_MIRROR:
        os.environ["HF_ENDPOINT"] = config.HF_MIRROR
        print(f"使用镜像源: {config.HF_MIRROR}")
    else:
        print("未配置Hugging Face镜像，将使用官方源")


def init_llm(config: Config):
    """初始化OpenAI兼容API的LLM模型"""
    # 确保provider是openai
    if config.LLM_MODEL.provider.lower() != "openai":
        raise ValueError(f"当前init_llm仅支持openai provider，实际为: {config.LLM_MODEL.provider}")

    # 设置环境变量
    os.environ["OPENAI_API_BASE"] = config.LLM_MODEL.apiBase
    os.environ["OPENAI_API_KEY"] = config.LLM_MODEL.apiKey

    # 根据模型类型选择初始化方式
    if config.LLM_MODEL.model_type.lower() == "chat":
        return ChatOpenAI(
            model_name=config.LLM_MODEL.model,
            temperature=config.LLM_MODEL.temperature,
            max_tokens=config.LLM_MODEL.max_tokens,
            openai_api_base=config.LLM_MODEL.apiBase,
            openai_api_key=config.LLM_MODEL.apiKey,
            request_timeout=120  # 增加超时设置，本地模型可能响应较慢
        )
    else:
        return OpenAI(
            model_name=config.LLM_MODEL.model,
            temperature=config.LLM_MODEL.temperature,
            max_tokens=config.LLM_MODEL.max_tokens,
            openai_api_base=config.LLM_MODEL.apiBase,
            openai_api_key=config.LLM_MODEL.apiKey,
            request_timeout=120
        )
