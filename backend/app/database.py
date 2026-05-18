"""数据库配置 - SQLite (用户数据)"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# 开发环境默认存项目根目录，生产环境通过 USER_DB_PATH 覆盖
_default_path = str(Path(__file__).parent.parent.parent / "data" / "users.db")
DB_PATH = os.environ.get("USER_DB_PATH", _default_path)
DATABASE_URL = f"sqlite:///{DB_PATH}"

# 确保数据目录存在
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
