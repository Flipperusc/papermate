"""Local username/password authentication for PaperMate."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
from typing import Any

from src.db import get_db_connection, init_db


PBKDF2_ALGORITHM = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 260_000
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.\-\u4e00-\u9fff]{3,32}$")
PASSWORD_MIN_LENGTH = 6


def normalize_username(username: str) -> str:
    """Return the canonical username used for login and display."""
    return re.sub(r"\s+", "", username or "").strip()


def validate_credentials(username: str, password: str) -> tuple[str, str]:
    """Validate registration/login input and return normalized values."""
    clean_username = normalize_username(username)
    clean_password = password or ""
    if not USERNAME_PATTERN.fullmatch(clean_username):
        raise ValueError("用户名需要 3-32 位，可包含中英文、数字、点、短横线和下划线。")
    if len(clean_password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"密码至少需要 {PASSWORD_MIN_LENGTH} 位。")
    return clean_username, clean_password


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256."""
    password_salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        password_salt,
        PBKDF2_ITERATIONS,
    )
    return "$".join(
        [
            PBKDF2_ALGORITHM,
            str(PBKDF2_ITERATIONS),
            password_salt.hex(),
            digest.hex(),
        ]
    )


def verify_password(password: str, stored_hash: str) -> bool:
    """Return whether a plain password matches the stored PBKDF2 hash."""
    try:
        algorithm, iterations_text, salt_hex, digest_hex = stored_hash.split("$", 3)
        iterations = int(iterations_text)
        salt = bytes.fromhex(salt_hex)
        expected_digest = bytes.fromhex(digest_hex)
    except (ValueError, TypeError):
        return False

    if algorithm != PBKDF2_ALGORITHM:
        return False

    actual_digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual_digest, expected_digest)


def create_user(username: str, password: str) -> dict[str, Any]:
    """Create a new local user account."""
    clean_username, clean_password = validate_credentials(username, password)
    init_db()

    try:
        with get_db_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (username, password_hash)
                VALUES (?, ?)
                """,
                (clean_username, hash_password(clean_password)),
            )
            user_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ValueError("这个用户名已经被占用，换个马甲试试。") from exc

    return {"user_id": user_id, "username": clean_username}


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    """Authenticate a local user and update the last-login timestamp."""
    clean_username = normalize_username(username)
    if not clean_username or not password:
        return None

    init_db()
    with get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT user_id, username, password_hash
            FROM users
            WHERE username = ? COLLATE NOCASE
            """,
            (clean_username,),
        ).fetchone()
        if not row or not verify_password(password, str(row["password_hash"])):
            return None

        connection.execute(
            """
            UPDATE users
            SET last_login_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
            """,
            (row["user_id"],),
        )

    return {"user_id": int(row["user_id"]), "username": str(row["username"])}


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    """Return a user record by id."""
    init_db()
    with get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT user_id, username, created_at, last_login_at
            FROM users
            WHERE user_id = ?
            """,
            (int(user_id),),
        ).fetchone()

    return dict(row) if row else None
