"""SqliteDatabase: the one place that opens connections and runs SQL."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator, Sequence

import pandas as pd

from medireach.config import check_sqlite_version
from medireach.errors import DataAccessError

IN_MEMORY = ":memory:"


def read_sql_file(path: str, name: str) -> str:
    """Read a .sql file; a missing or unreadable file is a DataAccessError named after it."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise DataAccessError(name, exc) from exc


class SqliteDatabase:
    """Runs named statements against one SQLite file and wraps every failure.

    Each statement gets a name so that a DataAccessError says what was being
    attempted, not just what SQLite complained about.
    """

    def __init__(self, db_path: str) -> None:
        check_sqlite_version()
        self.db_path = db_path
        # An in-memory database lives only as long as its connection, so that
        # one is kept open. File databases get a fresh connection per call.
        self._shared: sqlite3.Connection | None = self._open() if db_path == IN_MEMORY else None

    def _open(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self.db_path, check_same_thread=False)
            # SQLite does not enforce foreign keys unless asked, per connection.
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.Error as exc:
            raise DataAccessError("open_database", exc) from exc

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self._shared is not None:
            yield self._shared
            return
        connection = self._open()
        try:
            yield connection
        finally:
            connection.close()

    def query(self, name: str, sql: str, params: Sequence[object] | dict[str, object] = ()) -> pd.DataFrame:
        """Run a SELECT and return the result as a DataFrame."""
        with self._connection() as connection:
            try:
                return pd.read_sql_query(sql, connection, params=params)
            except (sqlite3.Error, pd.errors.DatabaseError) as exc:
                raise DataAccessError(name, exc) from exc

    def run_script(self, name: str, sql_text: str) -> None:
        """Run a multi-statement script such as schema.sql or views.sql."""
        with self._connection() as connection:
            try:
                connection.executescript(sql_text)
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise DataAccessError(name, exc) from exc

    def write(self, name: str, operations: Sequence[tuple[str, Sequence[tuple]]]) -> None:
        """Run several parameterised inserts as one transaction: all or nothing."""
        with self._connection() as connection:
            try:
                for sql, rows in operations:
                    connection.executemany(sql, rows)
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise DataAccessError(name, exc) from exc

    def close(self) -> None:
        if self._shared is not None:
            self._shared.close()
            self._shared = None
