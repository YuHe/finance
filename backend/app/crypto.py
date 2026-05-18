"""API Key 加解密工具（Fernet 对称加密）"""

import os
import hashlib
import base64
from cryptography.fernet import Fernet

_SECRET = os.environ.get("SECRET_KEY", "etf-rotation-secret-key-change-in-production")


def _get_fernet() -> Fernet:
    key_bytes = hashlib.sha256(_SECRET.encode()).digest()
    key_b64 = base64.urlsafe_b64encode(key_bytes)
    return Fernet(key_b64)


def encrypt_api_key(plain_key: str) -> str:
    return _get_fernet().encrypt(plain_key.encode()).decode()


def decrypt_api_key(encrypted_key: str) -> str:
    return _get_fernet().decrypt(encrypted_key.encode()).decode()
