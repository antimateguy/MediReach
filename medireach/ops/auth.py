"""PasswordHasher and AuthService: demo-grade login with salted hashes and an audit trail."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timezone
from typing import Callable

from medireach.domain import Event, Role, User
from medireach.errors import AccessDenied
from medireach.ops.repositories import EventLog, StoredCredentials, UserRepository

# Seeded demo accounts: (username, display name, role).
DEMO_USERS: tuple[tuple[str, str, Role], ...] = (
    ("admin", "Clinic operations manager", Role.ADMIN),
    ("reception1", "Receptionist 1", Role.RECEPTIONIST),
    ("reception2", "Receptionist 2", Role.RECEPTIONIST),
)


class PasswordHasher:
    """Salted PBKDF2-SHA256 from the standard library. A password is never stored or logged."""

    def __init__(self, iterations: int = 600_000) -> None:
        self._iterations = iterations

    def _derive(self, password: str, salt: str, iterations: int) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations).hex()

    def hash(self, password: str) -> StoredCredentials:
        salt = secrets.token_hex(16)
        return StoredCredentials(self._derive(password, salt, self._iterations), salt, self._iterations)

    def verify(self, password: str, stored: StoredCredentials) -> bool:
        candidate = self._derive(password, stored.salt, stored.iterations)
        # Constant-time comparison, so timing does not reveal how close a guess was.
        return hmac.compare_digest(candidate, stored.password_hash)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AuthService:
    """Logs people in, seeds the demo accounts, and checks roles."""

    def __init__(self, users: UserRepository, hasher: PasswordHasher, events: EventLog,
                 clock: Callable[[], str] = _utc_now) -> None:
        self._users = users
        self._hasher = hasher
        self._events = events
        self._clock = clock

    def seed_demo_users(self, password: str) -> list[User]:
        """Create the demo accounts that do not exist yet. Existing accounts are left alone."""
        created = []
        for username, display_name, role in DEMO_USERS:
            if self._users.find(username) is None:
                created.append(self._users.add(username, display_name, role, self._hasher.hash(password)))
        return created

    def login(self, username: str, password: str) -> User | None:
        """The user if the password is right and the account is active; otherwise None."""
        found = self._users.find(username.strip())
        ok = found is not None and found[0].active and self._hasher.verify(password, found[1])
        # Only the username is recorded, never the password that was typed.
        self._events.append(Event(self._clock(), username.strip() or "(blank)",
                                  "LOGIN_SUCCEEDED" if ok else "LOGIN_FAILED", "user", username.strip(), {}))
        return found[0] if ok and found is not None else None


def require_role(user: User | None, *roles: Role) -> User:
    """Return the user if they hold one of `roles`; raise AccessDenied otherwise."""
    if user is None or not user.active or user.role not in roles:
        allowed = ", ".join(role.value for role in roles)
        raise AccessDenied(f"this page needs one of these roles: {allowed}")
    return user
