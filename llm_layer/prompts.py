"""情绪分析 Prompt 模板"""

from data_layer.etf_pool import ETF_POOL

_ETF_INFO = {etf["code"]: etf for etf in ETF_POOL}

SENTIMENT_SYSTEM_PROMPT = """你是一位专业的A股行业ETF分析师。你需要分析指定行业/ETF当前的市场情绪和基本面状况。

请综合以下维度分析：
1. 近期该行业的政策面（利好/利空政策、监管动向）
2. 资金面（北向资金流向、主力资金动向、融资融券变化）
3. 基本面（行业景气度、业绩预期、估值水平）
4. 市场情绪（舆论热度、机构观点分歧）
5. 近期重要事件或催化剂"""

SENTIMENT_USER_TEMPLATE = """请搜索并分析 {etf_name}({etf_code}) 所属行业「{industry}」当前的市场情绪。

要求：
1. 综合政策面、资金面、基本面、情绪面给出分析（200字内）
2. 最后一行必须严格输出：SCORE: <数值>
   - 范围 -1.0 到 1.0
   - -1.0 表示极度看空，0 表示中性，1.0 表示极度看多
   - 仅在有充分证据时才给出极端值（>0.5 或 <-0.5）

今天: {today}"""


def build_sentiment_messages(code: str, today: str) -> list[dict]:
    """构建单只 ETF 的情绪分析 prompt"""
    info = _ETF_INFO.get(code, {})
    name = info.get("name", code)
    industry = info.get("industry", "未知")

    return [
        {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
        {"role": "user", "content": SENTIMENT_USER_TEMPLATE.format(
            etf_name=name, etf_code=code, industry=industry, today=today
        )},
    ]
