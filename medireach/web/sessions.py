"""SessionSigner: a signed, expiring login token, sent as `Authorization: Bearer` (standard library only)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import time


class SessionSigner:
    """Signs "username|expiry" with HMAC-SHA256. Nothing is stored on the server.

    The token only says who signed in; the role and the active flag are read
    from the user table on every request, so a changed account takes effect at once.
    """

    def __init__(self, secret: bytes, max_age_seconds: int = 12 * 3600, clock=time.time) -> None:
        if len(secret) < 16:
            raise ValueError("the session secret must be at least 16 bytes")
        self._secret = secret
        self.max_age = max_age_seconds
        self._clock = clock

    def _signature(self, payload: str) -> str:
        digest = hmac.new(self._secret, payload.encode("utf-8"), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    def sign(self, username: str) -> str:
        payload = f"{username}|{int(self._clock()) + self.max_age}"
        encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
        return f"{encoded}.{self._signature(payload)}"

    def username(self, token: str | None) -> str | None:
        """The signed-in username, or None for a missing, forged or expired token."""
        if not token or "." not in token:
            return None
        encoded, signature = token.rsplit(".", 1)
        try:
            payload = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8")
            username, expiry = payload.rsplit("|", 1)
            expires_at = int(expiry)
        except (ValueError, UnicodeDecodeError):
            return None
        if not hmac.compare_digest(signature, self._signature(payload)) or expires_at < self._clock():
            return None
        return username
