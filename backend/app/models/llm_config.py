"""LLM 配置与情绪缓存模型"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, Index
from app.database import Base


class LLMConfig(Base):
    __tablename__ = "llm_configs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    provider = Column(String(50), nullable=False)  # "dashscope" | "deepseek" | "custom"
    api_key_encrypted = Column(Text, nullable=False)
    api_base = Column(String(500), nullable=True)  # 自定义时填写
    model_name = Column(String(100), nullable=False)
    web_search_enabled = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_llm_configs_user_active", "user_id", "is_active"),
    )


class SentimentCache(Base):
    __tablename__ = "sentiment_cache"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(10), nullable=False)
    date = Column(String(10), nullable=False)
    score = Column(Float, nullable=False)  # -1.0 ~ 1.0
    raw_text = Column(Text, nullable=True)
    model_used = Column(String(100), nullable=True)
    user_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_sentiment_code_date_user", "code", "date", "user_id", unique=True),
    )
