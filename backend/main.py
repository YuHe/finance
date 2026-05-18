"""FastAPI 后端 - A股行业ETF轮动系统"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine
from app.models.user import User  # noqa: ensure table is registered
from app.database import Base
from app.routers import backtest, signal, portfolio, market
from app.routers import auth
from app.deps import get_current_user

# 创建所有表（用户表）
Base.metadata.create_all(bind=engine)

app = FastAPI(title="ETF轮动系统", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://finance.iwill.cc"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 认证路由（无需鉴权）
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])

# 业务路由（需要登录）
_auth_dep = [Depends(get_current_user)]
app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["backtest"], dependencies=_auth_dep)
app.include_router(signal.router, prefix="/api/v1/signal", tags=["signal"], dependencies=_auth_dep)
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"], dependencies=_auth_dep)
app.include_router(market.router, prefix="/api/v1/market", tags=["market"], dependencies=_auth_dep)


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}
