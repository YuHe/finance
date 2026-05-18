"""LLM 配置与情绪分析 API"""

import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models.user import User
from app.models.llm_config import LLMConfig, SentimentCache
from app.crypto import encrypt_api_key, decrypt_api_key

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_layer.providers import PROVIDERS, call_llm, LLMRequest
from llm_layer.llm_engine import run_sentiment_analysis, get_cached_sentiments_with_text

router = APIRouter()


# --- Schemas ---

class LLMConfigIn(BaseModel):
    provider: str  # "dashscope" | "deepseek" | "custom"
    api_key: str
    api_base: Optional[str] = None
    model_name: str
    web_search_enabled: bool = False


class LLMConfigOut(BaseModel):
    id: int
    provider: str
    api_base: Optional[str]
    model_name: str
    web_search_enabled: bool
    is_active: bool
    has_api_key: bool


# --- Endpoints ---

@router.get("/providers")
def list_providers():
    """列出预制提供商及其模型、能力"""
    result = []
    for pid, info in PROVIDERS.items():
        result.append({
            "id": pid,
            "name": info["name"],
            "api_base": info["api_base"],
            "models": info["models"],
            "supports_web_search": info["supports_web_search"],
        })
    return result


@router.get("/config")
def get_config(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """获取当前用户的 LLM 配置"""
    config = db.query(LLMConfig).filter(
        LLMConfig.user_id == user.id,
        LLMConfig.is_active == True,
    ).first()
    if not config:
        return None
    return LLMConfigOut(
        id=config.id,
        provider=config.provider,
        api_base=config.api_base,
        model_name=config.model_name,
        web_search_enabled=config.web_search_enabled,
        is_active=config.is_active,
        has_api_key=bool(config.api_key_encrypted),
    )


@router.put("/config")
def save_config(req: LLMConfigIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """保存/更新 LLM 配置"""
    # 如果 api_key 是占位符，复用旧密钥
    old_config = db.query(LLMConfig).filter(
        LLMConfig.user_id == user.id, LLMConfig.is_active == True
    ).first()

    if req.api_key == "__KEEP__":
        if not old_config:
            return {"success": False, "data": None, "error": {"code": "NO_KEY", "message": "无已保存的 API Key"}}
        encrypted_key = old_config.api_key_encrypted
    else:
        encrypted_key = encrypt_api_key(req.api_key)

    # 停用旧配置
    db.query(LLMConfig).filter(LLMConfig.user_id == user.id).update({"is_active": False})

    config = LLMConfig(
        user_id=user.id,
        provider=req.provider,
        api_key_encrypted=encrypted_key,
        api_base=req.api_base,
        model_name=req.model_name,
        web_search_enabled=req.web_search_enabled,
        is_active=True,
    )
    db.add(config)
    db.commit()
    db.refresh(config)
    return {"success": True, "data": {"id": config.id}, "error": None}


@router.post("/test")
def test_connection(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """测试 LLM 连接"""
    config = db.query(LLMConfig).filter(
        LLMConfig.user_id == user.id,
        LLMConfig.is_active == True,
    ).first()
    if not config:
        return {"success": False, "data": None, "error": {"code": "NO_CONFIG", "message": "未配置LLM"}}

    api_key = decrypt_api_key(config.api_key_encrypted)
    api_base = config.api_base or PROVIDERS.get(config.provider, {}).get("api_base", "")

    req = LLMRequest(
        messages=[{"role": "user", "content": "回复OK即可，不要其他内容"}],
        model=config.model_name,
        api_base=api_base,
        api_key=api_key,
        max_tokens=20,
    )
    result = asyncio.run(call_llm(req))
    if result.success:
        return {"success": True, "data": {"message": f"连接成功: {result.content[:50]}"}, "error": None}
    else:
        return {"success": False, "data": None, "error": {"code": "CONNECTION_FAILED", "message": result.error}}


@router.post("/analyze")
def trigger_analysis(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """手动触发全池情绪分析"""
    results = run_sentiment_analysis(user.id, db)
    return {"success": True, "data": {"count": len(results), "results": results}, "error": None}


@router.get("/sentiment")
def get_sentiment(date: Optional[str] = None, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """获取情绪分析结果"""
    from datetime import date as dt_date
    if date:
        # 查指定日期
        rows = db.query(SentimentCache).filter(
            SentimentCache.date == date,
            SentimentCache.user_id == user.id,
        ).all()
    else:
        rows = db.query(SentimentCache).filter(
            SentimentCache.date == dt_date.today().isoformat(),
            SentimentCache.user_id == user.id,
        ).all()

    data = [
        {"code": r.code, "score": r.score, "raw_text": r.raw_text,
         "model_used": r.model_used, "date": r.date}
        for r in rows
    ]
    return {"success": True, "data": data, "error": None}
