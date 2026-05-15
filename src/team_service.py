"""Team, project, and role helpers for PaperMate."""

from __future__ import annotations

import re
from typing import Any

from src.db import get_db_connection, init_db


ROLE_ORDER = {
    "viewer": 10,
    "editor": 20,
    "admin": 30,
    "owner": 40,
}
TEAM_ROLES = tuple(ROLE_ORDER.keys())
DEFAULT_PROJECT_NAME = "默认项目"


def normalize_team_name(name: str) -> str:
    """Return a safe team or project name."""
    clean_name = re.sub(r"\s+", " ", str(name or "")).strip()
    if not clean_name:
        raise ValueError("名称不能为空。")
    if len(clean_name) > 40:
        raise ValueError("名称最多 40 个字符。")
    return clean_name


def normalize_role(role: str) -> str:
    """Normalize and validate a team role."""
    clean_role = str(role or "").strip().lower()
    if clean_role not in ROLE_ORDER:
        raise ValueError("不支持的团队角色。")
    return clean_role


def role_at_least(role: str | None, minimum: str) -> bool:
    """Return whether role satisfies the required minimum role."""
    return ROLE_ORDER.get(str(role or "").lower(), 0) >= ROLE_ORDER[minimum]


def can_manage_team(role: str | None) -> bool:
    """Return whether role can manage teams, projects, and members."""
    return role_at_least(role, "admin")


def can_write(role: str | None) -> bool:
    """Return whether role can upload, enqueue jobs, ask, and save cards."""
    return role_at_least(role, "editor")


def ensure_user_workspace(user_id: int) -> dict[str, Any]:
    """Ensure a user has at least one team and one project."""
    init_db()
    with get_db_connection() as connection:
        membership = connection.execute(
            """
            SELECT t.team_id, t.name, m.role
            FROM team_members m
            JOIN teams t ON t.team_id = m.team_id
            WHERE m.user_id = ?
            ORDER BY
                CASE m.role
                    WHEN 'owner' THEN 1
                    WHEN 'admin' THEN 2
                    WHEN 'editor' THEN 3
                    ELSE 4
                END,
                t.team_id ASC
            LIMIT 1
            """,
            (int(user_id),),
        ).fetchone()
        if membership:
            team_id = int(membership["team_id"])
            project = ensure_default_project_for_connection(connection, team_id, user_id)
            return {
                "team_id": team_id,
                "team_name": str(membership["name"]),
                "role": str(membership["role"]),
                "project_id": int(project["project_id"]),
                "project_name": str(project["name"]),
            }

        username_row = connection.execute(
            "SELECT username FROM users WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()
        username = str(username_row["username"]) if username_row else "用户"
        cursor = connection.execute(
            """
            INSERT INTO teams (name, owner_user_id)
            VALUES (?, ?)
            """,
            (f"{username} 的团队", int(user_id)),
        )
        team_id = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT INTO team_members (team_id, user_id, role)
            VALUES (?, ?, 'owner')
            """,
            (team_id, int(user_id)),
        )
        project = ensure_default_project_for_connection(connection, team_id, user_id)
        return {
            "team_id": team_id,
            "team_name": f"{username} 的团队",
            "role": "owner",
            "project_id": int(project["project_id"]),
            "project_name": str(project["name"]),
        }


def ensure_default_project_for_connection(connection, team_id: int, user_id: int | None = None) -> dict[str, Any]:
    """Ensure a default project exists using an existing SQLite connection."""
    row = connection.execute(
        """
        SELECT project_id, team_id, name
        FROM projects
        WHERE team_id = ?
        ORDER BY project_id ASC
        LIMIT 1
        """,
        (int(team_id),),
    ).fetchone()
    if row:
        return dict(row)

    cursor = connection.execute(
        """
        INSERT INTO projects (team_id, name, created_by_user_id)
        VALUES (?, ?, ?)
        """,
        (int(team_id), DEFAULT_PROJECT_NAME, user_id),
    )
    return {
        "project_id": int(cursor.lastrowid),
        "team_id": int(team_id),
        "name": DEFAULT_PROJECT_NAME,
    }


def list_user_teams(user_id: int) -> list[dict[str, Any]]:
    """Return teams visible to a user."""
    init_db()
    ensure_user_workspace(user_id)
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                t.team_id,
                t.name,
                t.owner_user_id,
                m.role,
                t.created_at,
                t.updated_at
            FROM team_members m
            JOIN teams t ON t.team_id = m.team_id
            WHERE m.user_id = ?
            ORDER BY t.updated_at DESC, t.team_id DESC
            """,
            (int(user_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def get_user_team_role(user_id: int, team_id: int) -> str | None:
    """Return a user's role in one team."""
    init_db()
    with get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT role
            FROM team_members
            WHERE user_id = ? AND team_id = ?
            """,
            (int(user_id), int(team_id)),
        ).fetchone()
    return str(row["role"]) if row else None


def require_team_role(user_id: int, team_id: int, minimum_role: str = "viewer") -> str:
    """Return the user's role or raise when access is not allowed."""
    role = get_user_team_role(user_id, team_id)
    if not role_at_least(role, minimum_role):
        raise PermissionError("当前用户没有执行该操作的团队权限。")
    return str(role)


def list_projects(user_id: int, team_id: int) -> list[dict[str, Any]]:
    """Return projects for a team visible to a user."""
    require_team_role(user_id, team_id, "viewer")
    init_db()
    with get_db_connection() as connection:
        ensure_default_project_for_connection(connection, team_id, user_id)
        rows = connection.execute(
            """
            SELECT project_id, team_id, name, created_by_user_id, created_at, updated_at
            FROM projects
            WHERE team_id = ?
            ORDER BY updated_at DESC, project_id DESC
            """,
            (int(team_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def get_project(project_id: int, user_id: int, team_id: int | None = None) -> dict[str, Any] | None:
    """Return one project if visible to user."""
    init_db()
    with get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT project_id, team_id, name, created_by_user_id, created_at, updated_at
            FROM projects
            WHERE project_id = ?
            """,
            (int(project_id),),
        ).fetchone()
    if not row:
        return None
    project = dict(row)
    if team_id is not None and int(project["team_id"]) != int(team_id):
        return None
    require_team_role(user_id, int(project["team_id"]), "viewer")
    return project


def create_team(user_id: int, name: str) -> dict[str, Any]:
    """Create a team and make the current user its owner."""
    clean_name = normalize_team_name(name)
    init_db()
    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO teams (name, owner_user_id)
            VALUES (?, ?)
            """,
            (clean_name, int(user_id)),
        )
        team_id = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT INTO team_members (team_id, user_id, role)
            VALUES (?, ?, 'owner')
            """,
            (team_id, int(user_id)),
        )
        ensure_default_project_for_connection(connection, team_id, user_id)
    return get_team(team_id, user_id) or {"team_id": team_id, "name": clean_name, "role": "owner"}


def get_team(team_id: int, user_id: int) -> dict[str, Any] | None:
    """Return one team visible to a user."""
    init_db()
    with get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT t.team_id, t.name, t.owner_user_id, m.role, t.created_at, t.updated_at
            FROM teams t
            JOIN team_members m ON m.team_id = t.team_id
            WHERE t.team_id = ? AND m.user_id = ?
            """,
            (int(team_id), int(user_id)),
        ).fetchone()
    return dict(row) if row else None


def create_project(user_id: int, team_id: int, name: str) -> dict[str, Any]:
    """Create a project inside a team."""
    require_team_role(user_id, team_id, "admin")
    clean_name = normalize_team_name(name)
    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO projects (team_id, name, created_by_user_id)
            VALUES (?, ?, ?)
            """,
            (int(team_id), clean_name, int(user_id)),
        )
        project_id = int(cursor.lastrowid)
    project = get_project(project_id, user_id, team_id)
    return project or {"project_id": project_id, "team_id": int(team_id), "name": clean_name}


def list_team_members(user_id: int, team_id: int) -> list[dict[str, Any]]:
    """Return team members when requester can see the team."""
    require_team_role(user_id, team_id, "viewer")
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                m.team_id,
                m.user_id,
                u.username,
                m.role,
                m.created_at,
                m.updated_at
            FROM team_members m
            JOIN users u ON u.user_id = m.user_id
            WHERE m.team_id = ?
            ORDER BY
                CASE m.role
                    WHEN 'owner' THEN 1
                    WHEN 'admin' THEN 2
                    WHEN 'editor' THEN 3
                    ELSE 4
                END,
                u.username COLLATE NOCASE
            """,
            (int(team_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def add_team_member_by_username(
    requester_user_id: int,
    team_id: int,
    username: str,
    role: str = "viewer",
) -> dict[str, Any]:
    """Add an existing user to a team by username."""
    require_team_role(requester_user_id, team_id, "admin")
    clean_role = normalize_role(role)
    if clean_role == "owner":
        raise ValueError("不能直接添加 owner 角色成员。")
    clean_username = re.sub(r"\s+", "", username or "").strip()
    with get_db_connection() as connection:
        user = connection.execute(
            """
            SELECT user_id, username
            FROM users
            WHERE username = ? COLLATE NOCASE
            """,
            (clean_username,),
        ).fetchone()
        if not user:
            raise ValueError("没有找到这个用户名。")
        connection.execute(
            """
            INSERT INTO team_members (team_id, user_id, role)
            VALUES (?, ?, ?)
            ON CONFLICT(team_id, user_id) DO UPDATE SET
                role = excluded.role,
                updated_at = CURRENT_TIMESTAMP
            """,
            (int(team_id), int(user["user_id"]), clean_role),
        )
    return {
        "team_id": int(team_id),
        "user_id": int(user["user_id"]),
        "username": str(user["username"]),
        "role": clean_role,
    }


def update_team_member_role(
    requester_user_id: int,
    team_id: int,
    target_user_id: int,
    role: str,
) -> None:
    """Change a member role."""
    require_team_role(requester_user_id, team_id, "admin")
    clean_role = normalize_role(role)
    with get_db_connection() as connection:
        target = connection.execute(
            """
            SELECT role
            FROM team_members
            WHERE team_id = ? AND user_id = ?
            """,
            (int(team_id), int(target_user_id)),
        ).fetchone()
        if not target:
            raise ValueError("没有找到这个团队成员。")
        if str(target["role"]) == "owner":
            raise ValueError("不能修改 owner 的角色。")
        if clean_role == "owner":
            raise ValueError("不能将成员改为 owner。")
        connection.execute(
            """
            UPDATE team_members
            SET role = ?, updated_at = CURRENT_TIMESTAMP
            WHERE team_id = ? AND user_id = ?
            """,
            (clean_role, int(team_id), int(target_user_id)),
        )


def remove_team_member(requester_user_id: int, team_id: int, target_user_id: int) -> None:
    """Remove a non-owner member from a team."""
    require_team_role(requester_user_id, team_id, "admin")
    with get_db_connection() as connection:
        target = connection.execute(
            """
            SELECT role
            FROM team_members
            WHERE team_id = ? AND user_id = ?
            """,
            (int(team_id), int(target_user_id)),
        ).fetchone()
        if not target:
            return
        if str(target["role"]) == "owner":
            raise ValueError("不能移除团队 owner。")
        connection.execute(
            """
            DELETE FROM team_members
            WHERE team_id = ? AND user_id = ?
            """,
            (int(team_id), int(target_user_id)),
        )
