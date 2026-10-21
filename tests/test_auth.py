"""v3 G2: password hashing, demo accounts, login, role checks, append-only audit trail."""
from __future__ import annotations

from typing import Iterator

import pytest

from medireach.data.database import IN_MEMORY, SqliteDatabase, read_sql_file
from medireach.domain import Event, Role, User
from medireach.errors import AccessDenied, DataAccessError
from medireach.ops.auth import DEMO_USERS, AuthService, PasswordHasher, require_role
from medireach.ops.repositories import SqliteEventLog, SqliteUserRepository

PASSWORD = "correct horse battery staple"
FAST = PasswordHasher(iterations=1_000)   # tests do not need the production work factor


@pytest.fixture()
def ops(settings) -> Iterator[SqliteDatabase]:
    database = SqliteDatabase(IN_MEMORY)
    database.run_script("create_ops_schema", read_sql_file(settings.ops_schema_path, "read_ops_schema_file"))
    yield database
    database.close()


@pytest.fixture()
def auth(ops: SqliteDatabase) -> AuthService:
    service = AuthService(SqliteUserRepository(ops), FAST, SqliteEventLog(ops), clock=lambda: "2026-10-22T04:45:00+00:00")
    service.seed_demo_users(PASSWORD)
    return service


def test_hash_round_trip_uses_a_fresh_salt_each_time() -> None:
    first, second = FAST.hash(PASSWORD), FAST.hash(PASSWORD)
    assert first.salt != second.salt and first.password_hash != second.password_hash
    assert FAST.verify(PASSWORD, first) and FAST.verify(PASSWORD, second)
    assert not FAST.verify("wrong", first)
    assert PasswordHasher().hash(PASSWORD).iterations == 600_000


def test_demo_accounts_are_seeded_once(ops: SqliteDatabase, auth: AuthService) -> None:
    users = SqliteUserRepository(ops)
    assert [user.username for user in users.list_users()] == sorted(name for name, _, _ in DEMO_USERS)
    assert auth.seed_demo_users("a different password") == []
    assert auth.login("admin", PASSWORD) is not None     # the existing password was kept


def test_login_returns_the_user_and_its_role(auth: AuthService) -> None:
    admin = auth.login("admin", PASSWORD)
    receptionist = auth.login(" reception1 ", PASSWORD)
    assert admin is not None and admin.role == Role.ADMIN
    assert receptionist is not None and receptionist.role == Role.RECEPTIONIST


def test_wrong_password_unknown_user_and_inactive_account_are_refused(ops: SqliteDatabase, auth: AuthService) -> None:
    assert auth.login("admin", "guess") is None
    assert auth.login("nobody", PASSWORD) is None
    ops.write("deactivate", [("UPDATE users SET active = 0 WHERE username = ?", [("reception2",)])])
    assert auth.login("reception2", PASSWORD) is None


def test_no_password_is_stored_or_logged(ops: SqliteDatabase, auth: AuthService) -> None:
    auth.login("admin", PASSWORD)
    auth.login("admin", "typo-password")
    dumped = " ".join(str(value) for table in ("users", "events")
                      for value in ops.query("dump", f"SELECT * FROM {table}").to_numpy().ravel())
    assert PASSWORD not in dumped and "typo-password" not in dumped
    events = SqliteEventLog(ops).list_events(entity="user", entity_id="admin")
    assert list(events["event_type"]) == ["LOGIN_SUCCEEDED", "LOGIN_FAILED"]


def test_role_checks() -> None:
    admin = User(1, "admin", "Admin", Role.ADMIN)
    receptionist = User(2, "reception1", "Receptionist 1", Role.RECEPTIONIST)
    assert require_role(admin, Role.ADMIN) is admin
    assert require_role(receptionist, Role.ADMIN, Role.RECEPTIONIST) is receptionist
    with pytest.raises(AccessDenied):
        require_role(receptionist, Role.ADMIN)
    with pytest.raises(AccessDenied):
        require_role(None, Role.RECEPTIONIST)
    with pytest.raises(AccessDenied):
        require_role(User(3, "old", "Old", Role.ADMIN, active=False), Role.ADMIN)


def test_audit_trail_is_append_only(ops: SqliteDatabase) -> None:
    log = SqliteEventLog(ops)
    log.append(Event("2026-10-22T05:00:00+00:00", "admin", "PLAN_APPROVED", "plan", "1", {"calls": 431}))
    with pytest.raises(DataAccessError):
        ops.write("edit_event", [("UPDATE events SET actor = 'someone else'", [()])])
    with pytest.raises(DataAccessError):
        ops.write("delete_event", [("DELETE FROM events", [()])])
    events = log.list_events()
    assert len(events) == 1 and events.iloc[0]["payload"] == '{"calls": 431}'


def test_usernames_are_unique(ops: SqliteDatabase, auth: AuthService) -> None:
    with pytest.raises(DataAccessError):
        SqliteUserRepository(ops).add("admin", "Second admin", Role.ADMIN, FAST.hash(PASSWORD))
