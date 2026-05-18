"""LLM 情绪分析引擎 - V1 过滤器模式

核心逻辑：
- 情绪因子不参与评分排名
- 仅在动量与情绪产生"强背离"时介入：
  - 动量 Top N + 情绪 < -0.3 → 降权 50%
  - 动量 Top N + 情绪 < -0.6 → 剔除，换下一个
"""

import re
import asyncio
from datetime import date
from typing import Optional

from .providers import call_llm, LLMRequest, PROVIDERS
from .prompts import build_sentiment_messages
from data_layer.etf_pool import ETF_POOL

# 背离阈值
WARN_THRESHOLD = -0.3   # 情绪 < -0.3 → 降权
REJECT_THRESHOLD = -0.6  # 情绪 < -0.6 → 剔除


def parse_score(text: str) -> float:
    """从 LLM 输出中解析 SCORE: <float>"""
    match = re.search(r"SCORE:\s*([-+]?\d*\.?\d+)", text)
    if match:
        score = float(match.group(1))
        return max(-1.0, min(1.0, score))
    return 0.0


def apply_sentiment_filter(
    top_codes: list[str],
    all_scores_df,
    sentiment_data: dict[str, float],
) -> tuple[list[str], list[dict]]:
    """
    对动量选出的 Top N 应用情绪过滤。

    参数:
        top_codes: 动量选出的 Top N 代码列表
        all_scores_df: 完整因子评分 DataFrame (factor_engine 输出)
        sentiment_data: {code: sentiment_score} 情绪数据

    返回:
        (filtered_codes, warnings)
        filtered_codes: 过滤后的代码列表（可能从后续候选补位）
        warnings: 警告信息列表 [{code, action, sentiment, reason}]
    """
    if not sentiment_data:
        return top_codes, []

    filtered = []
    warnings = []
    rejected_count = 0

    for code in top_codes:
        sentiment = sentiment_data.get(code, 0.0)

        if sentiment < REJECT_THRESHOLD:
            # 剔除
            warnings.append({
                "code": code,
                "action": "rejected",
                "sentiment": sentiment,
                "reason": f"情绪极度看空({sentiment:.2f})，与动量背离，剔除",
            })
            rejected_count += 1
        elif sentiment < WARN_THRESHOLD:
            # 保留但标记警告（权重由 selector 降低）
            filtered.append(code)
            warnings.append({
                "code": code,
                "action": "warn",
                "sentiment": sentiment,
                "reason": f"情绪偏空({sentiment:.2f})，建议降低仓位",
            })
        else:
            # 正常
            filtered.append(code)

    # 补位：从排名表中找下一个没被选中的
    if rejected_count > 0 and all_scores_df is not None:
        remaining = all_scores_df[~all_scores_df["code"].isin(filtered)]["code"].tolist()
        for candidate in remaining:
            if len(filtered) >= len(top_codes):
                break
            cand_sentiment = sentiment_data.get(candidate, 0.0)
            if cand_sentiment >= REJECT_THRESHOLD:
                filtered.append(candidate)

    return filtered, warnings


def get_cached_sentiments(user_id: int, db) -> dict[str, float]:
    """获取今日缓存的情绪评分（用于因子过滤）"""
    from app.models.llm_config import SentimentCache
    today_str = date.today().isoformat()
    rows = db.query(SentimentCache).filter(
        SentimentCache.date == today_str,
        SentimentCache.user_id == user_id,
    ).all()
    return {row.code: row.score for row in rows}


def get_cached_sentiments_with_text(user_id: int, db) -> list[dict]:
    """获取今日缓存（含原始分析文本，展示用）"""
    from app.models.llm_config import SentimentCache
    today_str = date.today().isoformat()
    rows = db.query(SentimentCache).filter(
        SentimentCache.date == today_str,
        SentimentCache.user_id == user_id,
    ).all()
    return [
        {"code": r.code, "score": r.score, "raw_text": r.raw_text,
         "model_used": r.model_used, "created_at": r.created_at.isoformat() if r.created_at else None}
        for r in rows
    ]


async def _analyze_single(code: str, api_key: str, api_base: str,
                           model: str, web_search: bool, today_str: str):
    """分析单只 ETF"""
    messages = build_sentiment_messages(code, today_str)
    req = LLMRequest(
        messages=messages,
        model=model,
        api_base=api_base,
        api_key=api_key,
        web_search=web_search,
    )
    response = await call_llm(req)
    if response.success:
        score = parse_score(response.content)
        return code, score, response.content
    else:
        return code, 0.0, f"[ERROR] {response.error}"


def run_sentiment_analysis(user_id: int, db) -> list[dict]:
    """
    对全池 ETF 执行情绪分析。每只 ETF 每天最多调用一次。
    同步接口（内部用 asyncio.run）。
    """
    from app.models.llm_config import LLMConfig, SentimentCache
    from app.crypto import decrypt_api_key

    today_str = date.today().isoformat()

    # 获取用户 LLM 配置
    config = db.query(LLMConfig).filter(
        LLMConfig.user_id == user_id,
        LLMConfig.is_active == True,
    ).first()

    if not config:
        return []

    api_key = decrypt_api_key(config.api_key_encrypted)
    api_base = config.api_base or PROVIDERS.get(config.provider, {}).get("api_base", "")
    model = config.model_name
    web_search = config.web_search_enabled

    # 检查已缓存的
    existing = db.query(SentimentCache.code).filter(
        SentimentCache.date == today_str,
        SentimentCache.user_id == user_id,
    ).all()
    existing_codes = {row[0] for row in existing}

    codes_to_analyze = [etf["code"] for etf in ETF_POOL if etf["code"] not in existing_codes]

    if codes_to_analyze:
        # 逐个调用避免速率限制
        async def _run():
            results = []
            for code in codes_to_analyze:
                result = await _analyze_single(code, api_key, api_base, model, web_search, today_str)
                results.append(result)
            return results

        results = asyncio.run(_run())

        # 写入缓存
        for code, score, raw_text in results:
            entry = SentimentCache(
                code=code, date=today_str, score=score,
                raw_text=raw_text, model_used=model, user_id=user_id,
            )
            # upsert: 先查后更新或插入
            existing_entry = db.query(SentimentCache).filter(
                SentimentCache.code == code,
                SentimentCache.date == today_str,
                SentimentCache.user_id == user_id,
            ).first()
            if existing_entry:
                existing_entry.score = score
                existing_entry.raw_text = raw_text
                existing_entry.model_used = model
            else:
                db.add(entry)
        db.commit()

    # 返回所有今日数据
    return get_cached_sentiments_with_text(user_id, db)
