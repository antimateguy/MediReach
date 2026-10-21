"""Repositories for the workflow database (ops.db): users and the audit trail."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from medireach.data.database import SqliteDatabase
from medireach.domain import Event, Role, User
from medireach.errors import DataAccessError


@dataclass(frozen=True)
class StoredCredentials:
    """What is kept to check a password. Only the auth service ever sees this."""

    password_hash: str
    salt: str
    iterations: int


class UserRepository(ABC):
    """Users and their stored credentials."""

    @abstractmethod
    def add(self, username: str, display_name: str, role: Role, credentials: StoredCredentials) -> User:
        """Create a user; usernames are unique."""

    @abstractmethod
    def find(self, username: str) -> tuple[User, StoredCredentials] | None:
        """The user and their credentials, or None if there is no such user."""

    @abstractmethod
    def list_users(self) -> list[User]:
        """Every user, ordered by username."""


class EventLog(ABC):
    """The append-only audit trail."""

    @abstractmethod
    def append(self, event: Event) -> None:
        """Add one event. Events are never changed or removed."""

    def append_many(self, events: list[Event]) -> None:
        """Add several events; implementations may do it in one transaction."""
        for event in events:
            self.append(event)

    @abstractmethod
    def list_events(self, entity: str | None = None, entity_id: str | None = None) -> pd.DataFrame:
        """Events, oldest first, optionally for one entity."""


def _user(row: pd.Series) -> User:
    return User(int(row["user_id"]), str(row["username"]), str(row["display_name"]),
                Role(row["role"]), bool(row["active"]))


class SqliteUserRepository(UserRepository):
    """Users stored in ops.db."""

    _COLUMNS = "user_id, username, display_name, role, active, password_hash, salt, iterations"

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def add(self, username: str, display_name: str, role: Role, credentials: StoredCredentials) -> User:
        sql = """
            INSERT INTO users (username, display_name, role, password_hash, salt, iterations)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        row = (username, display_name, role.value, credentials.password_hash, credentials.salt, credentials.iterations)
        self._database.write("insert_user", [(sql, [row])])
        found = self.find(username)
        if found is None:
            raise DataAccessError("insert_user", f"user {username!r} was not stored")
        return found[0]

    def find(self, username: str) -> tuple[User, StoredCredentials] | None:
        sql = f"SELECT {self._COLUMNS} FROM users WHERE username = ?"
        frame = self._database.query("select_user", sql, (username,))
        if frame.empty:
            return None
        row = frame.iloc[0]
        return _user(row), StoredCredentials(str(row["password_hash"]), str(row["salt"]), int(row["iterations"]))

    def list_users(self) -> list[User]:
        frame = self._database.query("select_users", f"SELECT {self._COLUMNS} FROM users ORDER BY username")
        return [_user(row) for _, row in frame.iterrows()]


class SqliteEventLog(EventLog):
    """Events stored in ops.db; database triggers refuse any update or delete."""

    _INSERT = "INSERT INTO events (at, actor, event_type, entity, entity_id, payload) VALUES (?, ?, ?, ?, ?, ?)"

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    @staticmethod
    def _row(event: Event) -> tuple[str, ...]:
        return (event.at, event.actor, event.event_type, event.entity, event.entity_id,
                json.dumps(event.payload, sort_keys=True))

    def append(self, event: Event) -> None:
        self._database.write("insert_event", [(self._INSERT, [self._row(event)])])

    def append_many(self, events: list[Event]) -> None:
        if events:
            self._database.write("insert_events", [(self._INSERT, [self._row(event) for event in events])])

    def list_events(self, entity: str | None = None, entity_id: str | None = None) -> pd.DataFrame:
        sql = """
            SELECT event_id, at, actor, event_type, entity, entity_id, payload FROM events
            WHERE (:entity IS NULL OR entity = :entity) AND (:entity_id IS NULL OR entity_id = :entity_id)
            ORDER BY event_id
        """
        return self._database.query("select_events", sql, {"entity": entity, "entity_id": entity_id})
